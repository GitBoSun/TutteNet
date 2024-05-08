CUDA_VISIBLE_DEVICES=0 ns-train instant-ngp-bounded --data data/blender/lego \
	--pipeline.model.background-color white \
	--experiment-name lego_or \
	--pipeline.model.shape_name lego \
	--steps-per-save 1000 \
	--max-num-iterations 20000 \
	--pipeline.model.dump_density True \
	blender-data \
