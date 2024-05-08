CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/dist_train.sh 4 \
	cfgs/smpl_models/coord_model.yaml \
	cfgs/dataset_configs/smpl/amass.yaml \
	cfgs/optimizers/adamW_stepwise_data500.yaml \
	--extra_tag image_tmp_or_2 \
	--sync_bn \
	--eval_with_train \
	--max_ckpt_save_num 2 \
	--eval_interval 20 \
	--train_interval 20 \
	# multi0.5_res512_data5k_l8n11_j0.1_ep2k
	# amass_gau1.5_data5k_l8n11_ep1k_rotate \
	# --extra_tag pose_gau1.5_2.0_data5k_l24n11_ep2k_pred_fea64 \