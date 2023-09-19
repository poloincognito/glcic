Currently working on `train_cn.py`.

At each ssh connection:
- git pull
- update data and checkpoint
Then execute:
python train_cn.py 96 10^C../data/train/ ../logs/checkpoints/ -info 10