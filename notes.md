Currently working on `train_cn.py`.

At each ssh connection:
- git pull
- update data and checkpoint
Then execute:
nohup python train_cn.py 96 100 C../data/train/ ../logs/checkpoints/ -info 10