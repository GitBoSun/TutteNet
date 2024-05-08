CUDA_VISIBLE_DEVICES=0 ns-export tsdf \
	--downscale-factor 2 \
    --batch-size 10 \
    --add_deformation True \
    --tutte_deform_path ./nerf_deformation_models/lego/new2_up_rot0.45/new2_up_rot0.45_step4000.pt \
    --shape_name lego \
    --output-dir ./tsdf/lego/deform_up_rot0.45 \
    --load-config ./outputs/lego_final_up0.45/instant-ngp-bounded/2024-04-18_145159/config.yml  \
    