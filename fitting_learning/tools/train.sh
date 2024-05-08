CUDA_VISIBLE_DEVICES=2 python train.py \
	cfgs/smpl_models/coord_model.yaml \
	cfgs/dataset_configs/smpl/amass.yaml \
	cfgs/optimizers/adamW_stepwise_data500.yaml \
	--extra_tag tmp \
	--eval_with_train \
	--max_ckpt_save_num 2 \
	# --fix_random_seed \
	# --extra_tag  new_rand5k_appendbn_l5_bs16_laploss0.001 \
	# --extra_tag 5k_bn_100_sig_l5_laploss_ep300 \
