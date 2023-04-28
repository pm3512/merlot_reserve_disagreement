#! /usr/bin/bash

cd finetune/tvqa
python siq_finetune.py -ckpt=/home/aobolens/ckpt/sim_2/ -ne 3 --run_name=sim
sleep 10
python siq_finetune.py -ckpt=/home/aobolens/ckpt/sim_2/ -ne 3 --run_name=sim
sleep 10
python siq_finetune.py -ckpt=/home/aobolens/ckpt/sim_2/ -ne 3 --run_name=sim
sleep 10
python siq_finetune.py -ckpt=/home/aobolens/ckpt/base.ckpt -ne 3 --run_name=no_pretrain
sleep 10
python siq_finetune.py -ckpt=/home/aobolens/ckpt/base.ckpt -ne 3 --run_name=no_pretrain
sleep 10
python siq_finetune.py -ckpt=/home/aobolens/ckpt/base.ckpt -ne 3 --run_name=no_pretrain
sleep 10
cd ../../pretrain
python train.py  configs/pretrain_base.yaml  --run_name=sim_3 --sim_threshold 1 --video_threshold 0.5 --audio_threshold 0.5 --reweight
sleep 10
cd ../finetune/tvqa
python siq_finetune.py -ckpt=/home/aobolens/ckpt/sim_3/ckpt_760560 -ne 3 --run_name=sim_3
sleep 10
python siq_finetune.py -ckpt=/home/aobolens/ckpt/sim_3/ckpt_760560 -ne 3 --run_name=sim_3
sleep 10
python siq_finetune.py -ckpt=/home/aobolens/ckpt/sim_3/ckpt_760560 -ne 3 --run_name=sim_3