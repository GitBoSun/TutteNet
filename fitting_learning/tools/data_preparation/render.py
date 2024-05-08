# Conventions

# In OpenGL convention, the perspective projection matrix (as implemented in, e.g., utils.projection() in our samples and glFrustum() in OpenGL) treats the view-space z as increasing towards the viewer. However, after multiplication by perspective projection matrix, the homogeneous clip-space coordinate z/w increases away from the viewer. Hence, a larger depth value in the rasterizer output tensor also corresponds to a surface further away from the viewer.

from typing import Tuple
import imageio
import numpy as np
import torch
from matplotlib import image

import nvdiffrast.torch as dr
from psbody.mesh import Mesh
torch.concat = torch.cat


def _warmup(glctx):
    # windows workaround for https://github.com/NVlabs/nvdiffrast/issues/59
    def tensor(*args, **kwargs):
        return torch.tensor(*args, device="cuda", **kwargs)

    pos = tensor(
        [[[-0.8, -0.8, 0, 1], [0.8, -0.8, 0, 1], [-0.8, 0.8, 0, 1]]],
        dtype=torch.float32,
    )
    tri = tensor([[0, 1, 2]], dtype=torch.int32)
    dr.rasterize(glctx, pos, tri, resolution=[256, 256])


def vertex_color_render(
    vertices: torch.Tensor,  # B,V,3,
    faces: torch.Tensor,  # V,3,
    vertices_color: torch.Tensor,  # V,3,
    uv: torch.Tensor,  # V,3,
    mv: torch.Tensor,  # B,4,4
    proj: torch.Tensor ,  # B,4,4
    image_size: Tuple[int, int],
    texture: torch.Tensor = None,  # B,H,W,3
    glctx: dr.RasterizeCudaContext = None,
) -> torch.Tensor:  # B,H,W,4
    # Number of vertices
    V = vertices.shape[1]
    B = vertices.shape[0]

    # put everything on the GPU
    device = vertices.device
    faces = faces.to("cuda:0").type(torch.int32)

    vertices = vertices.to("cuda:0").float()
    # uv = uv.to("cuda:0").float()
    if vertices_color is not None:
        colors = vertices_color.to("cuda:0").float()

    # mv = mv.to("cuda:0").float()
    # proj = proj.to("cuda:0").float()

    if texture is not None:
        texture = texture.to("cuda:0").float().contiguous()

    # Change the type of faces to int32
    faces = faces.type(torch.int32)

    # Add a homogeneous coordinate to the vertices
    vert_hom = torch.cat((vertices, torch.ones(B, V, 1, device=vertices.device)), axis=-1)  # V,3 -> V,4c

    # Transform the vertices to clip space
    # vertices_clip = vert_hom @ mv.transpose(-2, -1) @ proj.transpose(-2, -1)  # C,V,4
    vertices_clip = vert_hom @ mv.transpose(-2, -1) 
    
    # orthographic
    # Change of convention to OPENGL CLIP Space. Z is pointing from the viewer into the screen
    vertices_clip[:, :, 2] = -vertices_clip[:, :, 2]
    # Memory convention in the rendered image. Either
    vertices_clip[:, :, 1] = -vertices_clip[:, :, 1]
    
    vertices_clip[:, :, 2] += 0.2
    # or
    # col = torch.flip(col, 1)

    # Check if gltctx is provided, otherwise create a new one
    if glctx is None:
        # glctx = dr.RasterizeGLContext()
        glctx = dr.RasterizeCudaContext(torch.device("cuda"))
        _warmup(glctx)
    
    # Rasterize data
    # print(vertices_clip.shape, faces.shape)
    rast_out, rast_out_db = dr.rasterize(glctx, vertices_clip, faces, resolution=image_size, grad_db=True)  # C,H,W,4

    if texture is None:
        # interpolate depth for debugging
        # col, _ = dr.interpolate(vertices_clip[0,:,2:3].contiguous().repeat(1,3).contiguous(), rast_out, faces)  # C,H,W,3
        col, _ = dr.interpolate(colors, rast_out, faces)  # C,H,W,3
        # create alpha channel
    else:
        texc, texd = dr.interpolate(uv.float(), rast_out, faces, rast_db=rast_out_db, diff_attrs="all")
        col = dr.texture(texture, texc, texd, filter_mode="linear-mipmap-linear", max_mip_level=9)
    alpha = torch.clamp(rast_out[..., -1:], max=1)  # C,H,W,1
    depth = rast_out[:, :, :, 2]  # C,H,W,1
    # if debugging with depth
    #  (col+1)/2*alpha + col*0*(1-alpha)

    # Add alpha channel
    col = torch.concat((col, alpha), dim=-1)  # C,H,W,4
    # Anti-aliasing
    col = dr.antialias(col, rast_out, vertices_clip, faces)  # C,H,W,4
    # col = col - col.min() / (col.max() - col.min())

    return col, depth  # C,H,W,4

if __name__=="__main__":
    
    mesh = Mesh(filename='/home/bos/projects/PCPerception/tools/template/smpl_template_leg_small.obj')
    pos = mesh.v 
    # alpha_x = np.random.rand() * 2 * np.pi
    alpha_y = np.random.rand() * 2 * np.pi
    # alpha_z = np.random.rand() * 2 * np.pi
    alpha_x =  np.pi/2
    # alpha_y =  np.pi/2
    alpha_z =  np.pi/2
    rot_x = np.array([[1, 0, 0], [0, np.cos(alpha_x), np.sin(alpha_x)], [0, -np.sin(alpha_x), np.cos(alpha_x)]])
    rot_y = np.array([[np.cos(alpha_y), 0, np.sin(alpha_y)], [0, 1, 0], [-np.sin(alpha_y), 0, np.cos(alpha_y)]])
    rot_z = np.array([[np.cos(alpha_z), np.sin(alpha_z), 0], [-np.sin(alpha_z), np.cos(alpha_z), 0], [0, 0, 1]])
    # pos = np.matmul(pos, rot_x)
    pos = np.matmul(pos, rot_y)
    # pos = np.matmul(pos, rot_z)

    tri = torch.from_numpy((mesh.f).astype(np.int32)).int().cuda()
    pos = torch.from_numpy(pos).float().cuda().unsqueeze(0)
    # pos = torch.cat((pos, torch.ones(pos.shape[0], 1).cuda()), dim=1).unsqueeze(0)
    col = torch.ones((pos.shape[0], pos.shape[1], 3), dtype=torch.float32).cuda()

    mv = np.eye(4)
    mv[1, 3] += 0.2
    mv = torch.from_numpy(mv).cuda().unsqueeze(0).float()
    
    color, depth = vertex_color_render(pos, tri, col[0], uv=None, mv=mv, proj=None, image_size=(256, 256))
    
    img_or = depth.cpu().numpy()[0] # Flip vertically.
    print(img_or.max(), img_or.min())
    # img =  img / 2
    # img = img_or - img_or.min() 
    img = img_or +1
    img =  img / 2
    img = 1- img
    
    img = img * (1-(img_or==0).astype(np.float32))
    img = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8) # Quantize to np.uint8
    imageio.imsave('depth.png', img)
    
    img = color.cpu().numpy()[0,] # Flip vertically.
    img = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8) # Quantize to np.uint8

    # print("Saving to 'tri.png'.")
    imageio.imsave('color.png', img)


    # import pdb; pdb.set_trace() 
