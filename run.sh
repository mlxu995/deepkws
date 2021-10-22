python=/home/mlxu/anaconda3/envs/speechbrain/bin/python
gpus=1
use_gpu=2
config=hparams/resnet15.yaml
# config=hparams/ds-res17.yaml

# CUDA_VISIBLE_DEVICES=${use_gpu} ${python} -m torch.distributed.launch --nproc_per_node=${gpus} train.py ${config} --distributed_launch --distributed_backend='nccl'
CUDA_VISIBLE_DEVICES=${use_gpu} ${python} train.py ${config} 
