CUDA_VISIBLE_DEVICES=0 ns-export tsdf \
	--downscale-factor 2 \
    --batch-size 10 \
    --output-dir ./tsdf/lego/or \
    --load-config ./outputs/lego/instant-ngp-bounded/2023-08-22_205042/config.yml \
