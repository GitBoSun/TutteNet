CUDA_VISIBLE_DEVICES=2 python render_data.py \
	cfgs/smpl_models/coord_model.yaml \
	cfgs/dataset_configs/smpl/amass.yaml \
	cfgs/optimizers/adamW_stepwise_data500.yaml \
	--extra_tag tmp_render2 \
	--eval_with_train \
	--max_ckpt_save_num 2 \
	--save_train false \
	--start_idx 10 \
	# 0, 1, 2, 3, 4, 5, 6
