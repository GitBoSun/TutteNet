import numpy as np


class PointFeatureEncoder(object):
    def __init__(self, config):
        super().__init__()
        self.point_encoding_config = config
        self.used_feature_list = self.point_encoding_config.used_feature_list
        self.src_feature_list = self.point_encoding_config.src_feature_list

    @property
    def num_point_features(self):
        return getattr(self, self.point_encoding_config.encoding_type)(point_feat=None)

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                ...
        Returns:
            data_dict:
                points: (N, 3 + C_out),
                use_lead_xyz: whether to use xyz as point-wise features
                ...
        """
        data_dict['point_feat'] = getattr(self, self.point_encoding_config.encoding_type)(
            data_dict['point_feat']
        )
        #data_dict['use_lead_xyz'] = use_lead_xyz
       
        #if self.point_encoding_config.get('filter_sweeps', False) and 'timestamp' in self.src_feature_list:
        #    max_sweeps = self.point_encoding_config.max_sweeps
        #    idx = self.src_feature_list.index('timestamp')
        #    dt = np.round(data_dict['points'][:, idx], 2)
        #    max_dt = sorted(np.unique(dt))[min(len(np.unique(dt))-1, max_sweeps-1)]
        #    data_dict['points'] = data_dict['points'][dt <= max_dt]
        
        return data_dict

    def absolute_coordinates_encoding(self, point_feat=None):
        if point_feat is None:
            num_output_features = len(self.used_feature_list)
            return num_output_features

        if len(self.used_feature_list) == 0:
            return point_feat[:, :0]

        point_feature_list = []
        for x in self.used_feature_list:
            idx = self.src_feature_list.index(x)
            point_feature_list.append(point_feat[:, idx:idx+1])
        point_feat = np.concatenate(point_feature_list, axis=1)
        
        return point_feat
