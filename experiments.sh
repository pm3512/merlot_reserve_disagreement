cd pretrain
#cd ../../pretrain
#python train.py  configs/pretrain_base.yaml  --run_name=social_iq_t1 --reweight --dataset social_iq --sim_threshold_text 100 --sim_threshold_audio 0 --sim_threshold_video 100
#python train.py  configs/pretrain_base.yaml  --run_name=mustard_allsim_17 --reweight --dataset mustard --sim_threshold_text 100 --sim_threshold_audio 100 --sim_threshold_video 100
#python train.py  configs/pretrain_base.yaml  --run_name=mustard_allsim_15 --reweight --dataset mustard --sim_threshold_text 0 --sim_threshold_audio 0 --sim_threshold_video 0
#python train.py  configs/pretrain_base.yaml  --run_name=social_iq_baseline --reweight --dataset social_iq --sim_threshold_text 100 --sim_threshold_audio 100 --sim_threshold_video 100
cd ../finetune/tvqa
#cd finetune/tvqa

for i in {0..50}
do
    python siq_finetune.py -ckpt=/home/aobolens/ckpt/mustard_allsim_17 -ne 5 --run_name=mustard_allsim_17 --dataset mustard
    python siq_finetune.py -ckpt=/home/aobolens/ckpt/mustard_allsim_15 -ne 5 --run_name=mustard_allsim_15 --dataset mustard
    #python siq_finetune.py -ckpt=/home/aobolens/ckpt/social_iq_t3 -ne 1 --run_name=social_iq_t3 --dataset social_iq
    #python siq_finetune.py -ckpt=/home/aobolens/ckpt/social_iq_baseline -ne 1 --run_name=social_iq_baseline --dataset social_iq
    rm -rf wandb
done
