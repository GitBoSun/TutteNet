from .tutte_head import TutteHead
from .mlp_head import MLPHead
# from .sdf_head import SDFHead
from .tutte_head_3d import TutteHead3D
from .tutte_head_3d_distortion import TutteHead3D_Distortion


__all__ = {
    'TutteHead': TutteHead,
    'MLPHead': MLPHead,
    # 'SDFHead': SDFHead, 
    'TutteHead3D': TutteHead3D, 
    'TutteHead3D_Distortion': TutteHead3D_Distortion, 
    }
