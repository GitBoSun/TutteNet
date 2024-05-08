# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Proposal network field.
"""

from typing import Literal, Optional, Tuple

import torch
from torch import Tensor, nn
import torch_cluster

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.encodings import HashEncoding
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field

from nerfstudio.tutte_modules.tutte_utils import density_to_tutte, tutte_to_density, get_points_scaling_data, density_to_learning, learning_to_density

class HashMLPDensityField(Field):
    """A lightweight density field module.

    Args:
        aabb: parameters of scene aabb bounds
        num_layers: number of hidden layers
        hidden_dim: dimension of hidden layers
        spatial_distortion: spatial distortion module
        use_linear: whether to skip the MLP and use a single linear layer instead
    """

    aabb: Tensor

    def __init__(
        self,
        aabb: Tensor,
        num_layers: int = 2,
        hidden_dim: int = 64,
        spatial_distortion: Optional[SpatialDistortion] = None,
        use_linear: bool = False,
        num_levels: int = 8,
        max_res: int = 1024,
        base_res: int = 16,
        log2_hashmap_size: int = 18,
        features_per_level: int = 2,
        implementation: Literal["tcnn", "torch"] = "torch",
        add_deformation: bool = False, 
        tutte_model_learning = None, 
        tutte_model=None,
        var_pred_torch=None, 
        shape_name="", 
        shape_points_cuda = None,
        batch_y = None,  
        keep_shape=False, 

    ) -> None:
        super().__init__()
        self.register_buffer("aabb", aabb)
        self.spatial_distortion = spatial_distortion
        self.use_linear = use_linear
        self.add_deformation = add_deformation
        self.tutte_model_learning = tutte_model_learning 
        self.var_pred_torch = var_pred_torch 
        self.tutte_model = tutte_model 
        self.shape_name = shape_name 
        self.keep_shape = keep_shape 
        self.shape_points_cuda = shape_points_cuda
        self.batch_y = batch_y
        self.points_scaling_data = get_points_scaling_data(self.shape_points_cuda)
        self.register_buffer("max_res", torch.tensor(max_res))
        self.register_buffer("num_levels", torch.tensor(num_levels))
        self.register_buffer("log2_hashmap_size", torch.tensor(log2_hashmap_size))

        self.encoding = HashEncoding(
            num_levels=num_levels,
            min_res=base_res,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size,
            features_per_level=features_per_level,
            implementation=implementation,
        )

        if not self.use_linear:
            network = MLP(
                in_dim=self.encoding.get_out_dim(),
                num_layers=num_layers,
                layer_width=hidden_dim,
                out_dim=1,
                activation=nn.ReLU(),
                out_activation=None,
                implementation=implementation,
            )
            self.mlp_base = torch.nn.Sequential(self.encoding, network)
        else:
            self.linear = torch.nn.Linear(self.encoding.get_out_dim(), 1)

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, None]:
        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(ray_samples.frustums.get_positions())
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
            
        positions_flat = positions.view(-1, 3)
        new_positions = positions_flat
        
        if self.add_deformation:
            new_positions = positions_flat.clone().cpu()
            new_positions = (new_positions - 0.5)*2
            # print('new density', positions.shape, new_positions.min(0)[0], new_positions.max(0)[0])

            bn = 50000
            num_b = new_positions.shape[0]//bn + 1
            new_positions_pred = torch.zeros(new_positions.shape[0], 3)
            # print('gg')
            for bi in range(num_b):
                if new_positions[bi*bn:(bi+1)*bn].shape[0]==0:
                    continue
                cur_points = new_positions[bi*bn:(bi+1)*bn].unsqueeze(0)
                if self.tutte_model_learning is not None:
                    learning_points = density_to_learning(cur_points[0]).float()
                    # print('density learning', learning_points.max(0)[0], learning_points.min(0)[0])
                    learning_points = torch.clamp(learning_points, -0.95, 0.95).unsqueeze(0)
                    learning_pred = self.tutte_model_learning(learning_points, self.var_pred_torch, inverse=True)
                    learning_pred_points = learning_pred['pred_points'][0]
                    cur_points = learning_to_density(learning_pred_points)
                    cur_points = cur_points.unsqueeze(0).float()
                    #cur_points = cur_points*bound_mask[...,None] + new_positions[bi*bn:(bi+1)*bn].unsqueeze(0) * (~bound_mask[...,None])
                elif self.tutte_model is not None:
                    cur_points = density_to_tutte(cur_points[0], self.points_scaling_data, self.shape_name).float().unsqueeze(0)
                    # print('density tutte', cur_points[0].max(0)[0], cur_points[0].min(0)[0])
                    cur_points = torch.clamp(cur_points, -0.95, 0.95)
                    tmp_pred = self.tutte_model(cur_points, inverse=True)
                    cur_points = tmp_pred['pred_points'][0]
                    cur_points = tutte_to_density(cur_points, self.points_scaling_data, self.shape_name)
                    cur_points = cur_points.unsqueeze(0).float()
                new_positions_pred[bi*bn:(bi+1)*bn] = cur_points[0]

            new_positions_pred = new_positions_pred / 2 + 0.5
            new_positions_pred = new_positions_pred.cuda()

            new_positions_pred = torch.clamp(new_positions_pred, 0, 1)
        else:
            new_positions_pred = new_positions.clone()
        
        if not self.use_linear:
            density_before_activation = (
                self.mlp_base(new_positions_pred).view(*ray_samples.frustums.shape, -1).to(positions)
            )
        else:
            x = self.encoding(new_positions_pred).to(positions)
            density_before_activation = self.linear(x).view(*ray_samples.frustums.shape, -1)

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation)
        density = density * selector[..., None]
        
        if self.keep_shape and self.shape_points_cuda is not None:
            batch_x = torch.zeros(new_positions_pred.shape[0]).cuda()
            nearest_ids = torch_cluster.radius(new_positions_pred, self.shape_points_cuda, r=0.003, \
                        batch_x=batch_x, batch_y=self.batch_y, max_num_neighbors=500)
            nearest_ids = torch.unique(nearest_ids[1])
            w, h = density.shape[0], density.shape[1]
            selector_shape = (density < 0).all(dim=-1).view(-1)
            selector_shape[nearest_ids] = True
            selector_shape = selector_shape.view(w, h)
            density = density * selector_shape[..., None]
        return density, None

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None) -> dict:
        return {}
