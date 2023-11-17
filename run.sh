CUDA_VISIBLE_DEVICES=0 nohup wandb agent benchoi93/STGCN_1114/rbzesmlm >> out0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup wandb agent benchoi93/STGCN_1114/rbzesmlm >> out1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup wandb agent benchoi93/STGCN_1114/rbzesmlm >> out2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup wandb agent benchoi93/STGCN_1114/rbzesmlm >> out3.log 2>&1 &