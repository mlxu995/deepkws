python=/home/mlxu/anaconda3/envs/speechbrain/bin/python
gpus=2
use_gpu=2,3
config=hparams/resnet15.yaml

CUDA_VISIBLE_DEVICES=${use_gpu} ${python} -m torch.distributed.launch --nproc_per_node=${gpus} train.py ${config} --distributed_launch --distributed_backend='nccl'