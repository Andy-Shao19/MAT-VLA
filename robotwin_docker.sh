docker run --gpus all \
           --shm-size=400g \
           -p 533:22 \
           --name sjn_rdt_eval \
           -v /mnt:/mnt \
           -v /shared_disk:/shared_disk \
           --restart always \
           -it \
           gigahubnew-cn-beijing.cr.volces.com/giga-tmp/wby_robotwin:v1 \
           /bin/bash