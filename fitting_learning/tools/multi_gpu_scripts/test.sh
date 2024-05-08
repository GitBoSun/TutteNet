bash scripts/dist_test.sh 2 \
	cfgs/smpl_models/coord_model.yaml \
	cfgs/dataset_configs/smpl/amass.yaml \
	cfgs/optimizers/adamW_stepwise_data500.yaml \
	--extra_tag image_gau1.5_2.0_data5k_l24n11_ep2k_pred_fea64 \
	--batch_size 128 \
	--eval_tag siming3_seq_martial3 \
	--vis_prefix siming3_ \
	--ckpt /home/bosun/projects/PCPerception/output/smpl_models/coord_model/smpl_amass/image_gau1.5_2.0_data5k_l24n11_ep2k_pred_fea64/ckpt/checkpoint_epoch_2000.pth \

	
	# multi0.5_res512_data5k_l8n11_j0.1_ep2k
	# shape_gau1.5_2.0_data5k_l24n11_ep2k_pred_gender_real

	# girl2_1: 37201, 33497, 1381, 14693, 36335, 5651, 14930, 32576, 35447, 25446, 15535,1355
	# robo1: 2042, 2411, 3077, 6988, 9790 
	# robo2: 415, 487, 962, 1367