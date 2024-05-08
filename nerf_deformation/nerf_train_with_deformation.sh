CUDA_VISIBLE_DEVICES=2 ns-train instant-ngp-bounded --data data/blender/lego \
	--pipeline.model.background-color white \
	--experiment-name lego_up \
	--steps-per-save 5 \
	--max-num-iterations 60 \
	--add-deformation True \
	--pipeline.model.add-deformation True \
	--pipeline.model.build_tutte True \
	--pipeline.model.tutte_deform_path ./nerf_deformation_models/lego/new2_up_rot0.45/new2_up_rot0.45_step4000.pt \
	--load-checkpoint ./outputs/lego/instant-ngp-bounded/2023-08-22_205042/nerfstudio_models/step-000029999.ckpt \
	blender-data \
