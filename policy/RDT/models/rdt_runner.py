import re, sys, os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import \
    DPMSolverMultistepScheduler

from pathlib import Path
# get current workspace
current_file = Path(__file__)
sys.path.append(os.path.join(current_file.parent))
from hub_mixin import CompatiblePyTorchModelHubMixin
from rdt.model import RDT
    
class TrajLabelEncoder(nn.Module):
    """Embeds 8×14 diff‑label tensor (values 0/1/2) into token seq.
    Adds index 3 as **[MASK]** so训练时可以随机遮盖某些段。
    """

    def __init__(self, lang_token_dim: int):
        super().__init__()
        self.axis_emb = nn.ModuleList([nn.Embedding(4, lang_token_dim) for _ in range(14)])  # 0/1/2/3(mask)
        self.seg_emb = nn.Embedding(8, lang_token_dim)  # segment positional
        nn.init.normal_(self.seg_emb.weight, std=0.02)

    def forward(self, label: torch.LongTensor):
        """
        label: (B, 8, 14) with ints in {0,1,2,3}; 3 = [MASK]
        return: (B, 112, hidden_size)
        """
        # Ensure label is integer indices and on the same device as embeddings
        if label.dtype != torch.long:
            label = label.long()
        emb_device = self.axis_emb[0].weight.device
        if label.device != emb_device:
            label = label.to(emb_device)

        B, S, D = label.shape  # S=8, D=14
        
        tokens = []
        for d in range(D):
            tok = self.axis_emb[d](label[:, :, d])  # (B, S, H)
            tokens.append(tok)
        tokens = torch.stack(tokens, dim=2)  # (B, S, 14, H)
        seg_pos = self.seg_emb.weight.to(tokens.device).unsqueeze(1)  # (8,1,H)
        tokens = tokens + seg_pos  # broadcast on segment dim
        tokens = tokens.view(B, -1, tokens.size(-1))  # (B,112,H)
        return tokens


# ============================================================
#   8-token  Segment-MLP  Trajectory-Label Encoder
# ============================================================


class _SegmentTokeniser(nn.Module):
    """把单段 14 维 {0,1,2,3(mask)} → one-hot 42 → MLP → token_d"""

    def __init__(self, token_d: int, mid_d: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(14 * 3, mid_d),
            nn.GELU(),
            nn.Linear(mid_d, token_d),
        )

    def forward(self, seg: torch.LongTensor):  # (B,14)
        # 3 = mask → 0 向量 (信息缺失)
        seg = seg.clamp_max(2)
        onehot = torch.nn.functional.one_hot(seg, num_classes=3).flatten(-2)  # (B,42)
        # 关键：将输入转换为与权重相同的 dtype（避免 Float vs BFloat16 冲突）
        in_dtype = self.mlp[0].weight.dtype
        onehot = onehot.to(in_dtype)
        return self.mlp(onehot)  # (B, token_d)


class TrajLabelEncoder(nn.Module):
    """8 段 ×14 离散标签 → 8 个 token，维度 = text hidden dim"""

    def __init__(self, token_d: int):
        super().__init__()
        self.seg_embed = nn.Embedding(8, token_d)
        self.tokeniser = _SegmentTokeniser(token_d)

    def forward(self, label: torch.LongTensor):  # (B,8,14)
        if label.dtype != torch.long:
            label = label.long()
        B, S, _ = label.shape
        out = []
        for s in range(S):
            tok = self.tokeniser(label[:, s, :])  # (B, token_d)
            out.append(tok)
        tokens = torch.stack(out, dim=1)  # (B,8,token_d)
        # 关键：与 embedding 的 dtype 对齐后再相加
        tokens = tokens.to(self.seg_embed.weight.dtype)
        tokens = tokens + self.seg_embed.weight.unsqueeze(0)  # pos-embed
        return tokens  # (B,8,token_d)


class RDTRunner(nn.Module,
                CompatiblePyTorchModelHubMixin,
                repo_url="https://huggingface.co/robotics-diffusion-transformer/rdt-1b"):
    def __init__(self,
                 *,
                 action_dim,
                 pred_horizon,
                 config,
                 lang_token_dim,
                 img_token_dim,
                 state_token_dim,
                 max_lang_cond_len,
                 img_cond_len,
                 lang_pos_embed_config=None,
                 img_pos_embed_config=None,
                 dtype=torch.bfloat16):
        super(RDTRunner, self).__init__()
        # Create diffusion model
        hidden_size = config['rdt']['hidden_size']
        self.model = RDT(
            output_dim=action_dim,
            horizon=pred_horizon,
            hidden_size=hidden_size,
            depth=config['rdt']['depth'],
            num_heads=config['rdt']['num_heads'],
            max_lang_cond_len=max_lang_cond_len,
            img_cond_len=img_cond_len,
            lang_pos_embed_config=lang_pos_embed_config,
            img_pos_embed_config=img_pos_embed_config,
            dtype=dtype,
        )
        # -------- diff‑label encoder (trainable, small) --------
        self.traj_label_encoder = TrajLabelEncoder(lang_token_dim)
        # 关键：将编码器权重转到与全局一致的 dtype
        self.traj_label_encoder.to(dtype)
        # 随机遮盖概率（每个段）
        self.label_mask_p = 0  # per‑slot mask prob (on 8×14 grid)
        
        # Create adpators for various conditional inputs
        self.lang_adaptor = self.build_condition_adapter(config['lang_adaptor'],
                                                         in_features=lang_token_dim,
                                                         out_features=hidden_size)
        self.img_adaptor = self.build_condition_adapter(config['img_adaptor'],
                                                        in_features=img_token_dim,
                                                        out_features=hidden_size)
        # A `state` refers to an action or a proprioception vector
        self.state_adaptor = self.build_condition_adapter(
            config['state_adaptor'],
            in_features=state_token_dim * 2,  # state + state mask (indicator)
            out_features=hidden_size)

        # Create the noise scheduler
        noise_scheduler_config = config['noise_scheduler']
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=noise_scheduler_config['num_train_timesteps'],
            beta_schedule=noise_scheduler_config['beta_schedule'],
            prediction_type=noise_scheduler_config['prediction_type'],
            clip_sample=noise_scheduler_config['clip_sample'],
        )
        self.noise_scheduler_sample = DPMSolverMultistepScheduler(
            num_train_timesteps=noise_scheduler_config['num_train_timesteps'],
            beta_schedule=noise_scheduler_config['beta_schedule'],
            prediction_type=noise_scheduler_config['prediction_type'],
        )

        self.num_train_timesteps = noise_scheduler_config['num_train_timesteps']
        self.num_inference_timesteps = noise_scheduler_config['num_inference_timesteps']
        self.prediction_type = noise_scheduler_config['prediction_type']

        self.pred_horizon = pred_horizon
        self.action_dim = action_dim

        print("Diffusion params: %e" %
              sum([p.numel() for p in self.model.parameters()] + [p.numel() for p in self.lang_adaptor.parameters()] +
                  [p.numel()
                   for p in self.img_adaptor.parameters()] + [p.numel() for p in self.state_adaptor.parameters()]))

    def build_condition_adapter(self, projector_type, in_features, out_features):
        projector = None
        if projector_type == 'linear':
            projector = nn.Linear(in_features, out_features)
        else:
            mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
            if mlp_gelu_match:
                mlp_depth = int(mlp_gelu_match.group(1))
                modules = [nn.Linear(in_features, out_features)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU(approximate="tanh"))
                    modules.append(nn.Linear(out_features, out_features))
                projector = nn.Sequential(*modules)

        if projector is None:
            raise ValueError(f'Unknown projector type: {projector_type}')

        return projector

    def adapt_conditions(self, lang_tokens, img_tokens, state_tokens):
        '''
        lang_tokens: (batch_size, lang_len, lang_token_dim)
        img_tokens: (batch_size, img_len, img_token_dim)
        state_tokens: (batch_size, state_len, state_token_dim)
        
        return: adpated (..., hidden_size) for all input tokens
        '''
        adpated_lang = self.lang_adaptor(lang_tokens)
        adpated_img = self.img_adaptor(img_tokens)
        adpated_state = self.state_adaptor(state_tokens)

        return adpated_lang, adpated_img, adpated_state

    def conditional_sample(self, lang_cond, lang_attn_mask, img_cond, state_traj, action_mask, ctrl_freqs):
        '''
        lang_cond: language conditional data, (batch_size, lang_len, hidden_size).
        lang_attn_mask: (batch_size, lang_len), a mask for valid language tokens,
            which should be True-False bool tensor.
        img_cond: image conditional data, (batch_size, img_len, hidden_size).
        state_traj: (batch_size, 1, hidden_size), state trajectory.
        action_mask: (batch_size, 1, action_dim), a 0-1 **float** tensor
            indicating the valid action dimensions.
        ctrl_freqs: (batch_size,), control frequency for each sample.
        
        return: (batch_size, horizon, action_dim)
        '''
        device = state_traj.device
        dtype = state_traj.dtype
        noisy_action = torch.randn(size=(state_traj.shape[0], self.pred_horizon, self.action_dim),
                                   dtype=dtype,
                                   device=device)
        action_mask = action_mask.expand(-1, self.pred_horizon, -1)

        # Set step values
        self.noise_scheduler_sample.set_timesteps(self.num_inference_timesteps)

        for t in self.noise_scheduler_sample.timesteps:
            # Prepare state-action trajectory
            action_traj = torch.cat([noisy_action, action_mask], dim=2)
            action_traj = self.state_adaptor(action_traj)
            state_action_traj = torch.cat([state_traj, action_traj], dim=1)

            # Predict the model output
            model_output = self.model(state_action_traj,
                                      ctrl_freqs,
                                      t.unsqueeze(-1).to(device),
                                      lang_cond,
                                      img_cond,
                                      lang_mask=lang_attn_mask)

            # Compute previous actions: x_t -> x_t-1
            noisy_action = self.noise_scheduler_sample.step(model_output, t, noisy_action).prev_sample
            noisy_action = noisy_action.to(state_traj.dtype)

        # Finally apply the action mask to mask invalid action dimensions
        noisy_action = noisy_action * action_mask

        return noisy_action

    # ========= Train  ============
    def compute_loss(self, lang_tokens, lang_attn_mask, img_tokens, state_tokens, action_gt, action_mask,
                     ctrl_freqs, traj_label=None) -> torch.Tensor:
        '''
        lang_tokens: (batch_size, lang_len, lang_token_dim)
        lang_attn_mask: (batch_size, lang_len), a mask for valid language tokens,
            which should be True-False bool tensor.
        img_tokens: (batch_size, img_len, img_token_dim)
        state_tokens: (batch_size, 1, state_token_dim)
        action_gt: (batch_size, horizon, state_token_dim), ground-truth actions for supervision
        action_mask: (batch_size, 1, state_token_dim), a 0-1 **float** tensor.
        ctrl_freqs: (batch_size,), control frequency for each sample.
        
        return: loss_value, a scalar tensor
        '''
        # ----- integrate diff‑label tokens into language tokens -----
        if traj_label is not None:
            # ---------- 随机 mask 部分段 ----------
            if self.training and self.label_mask_p > 0.0:
                B = traj_label.size(0)
                mask_choice = torch.rand(B, 8, 14, device=traj_label.device) < self.label_mask_p
                traj_label = traj_label.clone()
                traj_label[mask_choice] = 3  # 3 = [MASK]

            traj_tok = self.traj_label_encoder(traj_label)  # (B,112,H)
            traj_mask = torch.ones(traj_tok.shape[:-1], dtype=lang_attn_mask.dtype, device=lang_attn_mask.device)
            lang_tokens = torch.cat([traj_tok, lang_tokens], dim=1)
            lang_attn_mask = torch.cat([traj_mask, lang_attn_mask], dim=1)

        batch_size = lang_tokens.shape[0]
        device = lang_tokens.device
        # Sample noise that we'll add to the actions
        noise = torch.randn(action_gt.shape, dtype=action_gt.dtype, device=device)
        # Sample random diffusion timesteps
        timesteps = torch.randint(0, self.num_train_timesteps, (batch_size, ), device=device).long()
        # Add noise to the clean actions according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_action = self.noise_scheduler.add_noise(action_gt, noise, timesteps)

        # Concatenate the state and action tokens to form the input sequence
        state_action_traj = torch.cat([state_tokens, noisy_action], dim=1)
        # Append the action mask to the input sequence
        action_mask = action_mask.expand(-1, state_action_traj.shape[1], -1)
        state_action_traj = torch.cat([state_action_traj, action_mask], dim=2)
        # Align the dimension with the hidden size
        lang_cond, img_cond, state_action_traj = self.adapt_conditions(lang_tokens, img_tokens, state_action_traj)
        # Predict the denoised result
        pred = self.model(state_action_traj, ctrl_freqs, timesteps, lang_cond, img_cond, lang_mask=lang_attn_mask)

        pred_type = self.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = action_gt
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")
        loss = F.mse_loss(pred, target)
        return loss

    # ========= Inference  ============
    def predict_action(self, lang_tokens, lang_attn_mask, img_tokens, state_tokens, action_mask, ctrl_freqs, traj_label=None):
        '''
        lang_tokens: (batch_size, lang_len, lang_token_dim)
        lang_attn_mask: (batch_size, lang_len), a mask for valid language tokens,
            which should be True-False bool tensor.
        img_tokens: (batch_size, img_len, img_token_dim)
        state_tokens: (batch_size, 1, state_token_dim)
        action_mask: (batch_size, 1, action_dim),
            which should be a 0-1 **float** tensor.
        ctrl_freqs: (batch_size,), control frequency for each sample.
        
        return: (batch_size, horizon, action_dim), predicted action sequence
        '''
        # Prepare the state and conditions
        state_tokens = torch.cat([state_tokens, action_mask], dim=2)

        if traj_label is not None:
            diff_tok = self.traj_label_encoder(traj_label)
            diff_mask = torch.ones(diff_tok.shape[:-1], dtype=lang_attn_mask.dtype, device=lang_attn_mask.device)
            lang_tokens = torch.cat([diff_tok, lang_tokens], dim=1)
            lang_attn_mask = torch.cat([diff_mask, lang_attn_mask], dim=1)

        lang_cond, img_cond, state_traj = self.adapt_conditions(lang_tokens, img_tokens, state_tokens)

        # Run sampling
        action_pred = self.conditional_sample(
            lang_cond,
            lang_attn_mask,
            img_cond,
            state_traj,
            action_mask,
            ctrl_freqs,
        )

        return action_pred

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.compute_loss(*args, **kwargs)
