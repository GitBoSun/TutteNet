CUDA_VISIBLE_DEVICES=2 python test.py \
	cfgs/smpl_models/coord_model.yaml \
	cfgs/dataset_configs/smpl/amass.yaml \
	cfgs/optimizers/adamW_stepwise_data500.yaml \
	--extra_tag image_gau1.5_2.0_data5k_l24n11_ep2k_pred_fea64 \
	--batch_size 128 \
	--eval_tag  shape54_fix \
	--vis_prefix shape54_ \
	--ckpt /home/bosun/projects/PCPerception/output/smpl_models/coord_model/smpl_amass/image_gau1.5_2.0_data5k_l24n11_ep2k_pred_fea64/ckpt/checkpoint_epoch_2000.pth \
