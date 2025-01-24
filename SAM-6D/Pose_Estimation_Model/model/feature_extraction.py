import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.model_zoo as model_zoo
from functools import partial
import timm.models.vision_transformer
from model_utils import (
    LayerNorm2d,
    interpolate_pos_embed,
    get_chosen_pixel_feats,
    sample_pts_feats
)



class ViT(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, **kwargs):
        super(ViT, self).__init__(**kwargs)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)

        out = []
        d = len(self.blocks)
        n = d // 4
        idx_nblock = [d-1, d-n-1, d-2*n-1, d-3*n-1]

        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            if idx in idx_nblock:
                out.append(self.norm(x))
        return out



class ViT_AE(nn.Module):
    def __init__(self, cfg,) -> None:
        super(ViT_AE, self).__init__()
        self.cfg = cfg
        self.vit_type = cfg.vit_type
        self.up_type = cfg.up_type
        self.embed_dim = cfg.embed_dim
        self.out_dim = cfg.out_dim
        self.use_pyramid_feat = cfg.use_pyramid_feat
        self.pretrained = cfg.pretrained

        if self.vit_type == 'vit_base':
            self.vit = ViT(
                    patch_size=16, embed_dim=self.embed_dim, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),)
        elif self.vit_type == 'vit_large':
            self.vit = ViT(
                    patch_size=16, embed_dim=self.embed_dim, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6), )
        else:
            assert False

        if self.use_pyramid_feat:
            nblock = 4
        else:
            nblock = 1

        if self.up_type == 'linear':
            self.output_upscaling = nn.Linear(self.embed_dim * nblock, 16 * self.out_dim, bias=True)
        elif self.up_type == 'deconv':
            self.output_upscaling = nn.Sequential(
                nn.ConvTranspose2d(self.embed_dim * nblock, self.out_dim*2, kernel_size=2, stride=2),
                LayerNorm2d(self.out_dim*2),
                nn.GELU(),
                nn.ConvTranspose2d(self.out_dim*2, self.out_dim, kernel_size=2, stride=2),
            )
        else:
            assert False

        if self.pretrained:
            vit_checkpoint = os.path.join('checkpoints', 'mae_pretrain_'+ self.vit_type +'.pth')
            if not os.path.isdir(vit_checkpoint):
                if not os.path.isdir('checkpoints'):
                    os.makedirs('checkpoints')
                model_zoo.load_url('https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_'+ self.vit_type +'.pth', 'checkpoints')

            checkpoint = torch.load(vit_checkpoint, map_location='cpu')
            print("load pre-trained checkpoint from: %s" % vit_checkpoint)
            checkpoint_model = checkpoint['model']
            state_dict = self.vit.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]
            # interpolate position embedding
            interpolate_pos_embed(self.vit, checkpoint_model)
            msg = self.vit.load_state_dict(checkpoint_model, strict=False)


    def forward(self, x):
        B,_,H,W = x.size()
        vit_outs = self.vit(x)
        cls_tokens = vit_outs[-1][:,0,:].contiguous()
        vit_outs = [l[:,1:,:].contiguous() for l in vit_outs]

        if self.use_pyramid_feat:
            x = torch.cat(vit_outs, dim=2)
        else:
            x = vit_outs[-1]

        if self.up_type == 'linear':
            x = self.output_upscaling(x).reshape(B,14,14,4,4,self.out_dim).permute(0,5,1,3,2,4).contiguous()
            x = x.reshape(B,-1,56,56)
            x = F.interpolate(x, (H,W), mode="bilinear", align_corners=False)
        elif self.up_type == 'deconv':
            x = x.transpose(1,2).reshape(B,-1,14,14)
            x = self.output_upscaling(x)
            x = F.interpolate(x, (H,W), mode="bilinear", align_corners=False)
        return x, cls_tokens




class ViTEncoder(nn.Module):
    def __init__(self, cfg, npoint=2048):
        super(ViTEncoder, self).__init__()
        self.npoint = npoint
        self.rgb_net = ViT_AE(cfg)

    def forward(self, end_points):
        rgb = end_points['rgb']
        rgb_choose = end_points['rgb_choose']
        dense_fm = self.get_img_feats(rgb, rgb_choose)
        dense_pm = end_points['pts']
        assert rgb_choose.size(1) == self.npoint

        if not self.training and 'dense_po' in end_points.keys() and 'dense_fo' in end_points.keys():
            dense_po = end_points['dense_po'].clone()
            dense_fo = end_points['dense_fo'].clone()

            # get center and scale
            center = dense_po.mean(dim=1, keepdim=True) #(B, 1, 3)
            radius = torch.norm(dense_po, dim=2).max(1)[0]

            # translate to origin and normalize
            dense_po = sphere_points(dense_po, center, radius)
            dense_pm = sphere_points(dense_pm, center, radius)            

        else:
            tem1_rgb = end_points['tem1_rgb']
            tem1_choose = end_points['tem1_choose']
            tem1_pts = end_points['tem1_pts']
            tem2_rgb = end_points['tem2_rgb']
            tem2_choose = end_points['tem2_choose']
            tem2_pts = end_points['tem2_pts']

            # get dense point of object
            dense_po = torch.cat([tem1_pts, tem2_pts], dim=1)

            # get center and scale
            center = dense_po.mean(dim=1, keepdim=True) #(B, 1, 3)
            radius = torch.norm(dense_po, dim=2).max(1)[0]

            # translate to origin and normaliz
            dense_po = sphere_points(dense_po, center, radius)
            dense_pm = sphere_points(dense_pm, center, radius)
            tem1_pts = sphere_points(tem1_pts, center, radius)
            tem2_pts = sphere_points(tem2_pts, center, radius)

            dense_po, dense_fo = self.get_obj_feats(
                [tem1_rgb, tem2_rgb],
                [tem1_pts, tem2_pts],
                [tem1_choose, tem2_choose]
            )

        return dense_pm, dense_fm, dense_po, dense_fo, center, radius

    def get_img_feats(self, img, choose):
        return get_chosen_pixel_feats(self.rgb_net(img)[0], choose)

    def get_obj_feats(self, tem_rgb_list, tem_pts_list, tem_choose_list, npoint=None):
        if npoint is None:
            npoint = self.npoint

        tem_feat_list =[]
        for tem, tem_choose in zip(tem_rgb_list, tem_choose_list):
            tem_feat_list.append(self.get_img_feats(tem, tem_choose))

        tem_pts = torch.cat(tem_pts_list, dim=1)
        tem_feat = torch.cat(tem_feat_list, dim=1)

        return sample_pts_feats(tem_pts, tem_feat, npoint)


def sphere_points(points, center, radius, eps=1e-6):
    """
    Translate and scale a batch of point clouds so that `center` is the new origin
    and `radius` is the new scaling factor.

    Args:
        points (torch.Tensor):  (B, N, 3), the batch of point clouds.
        center (torch.Tensor):  (B, 1, 3), the centers used for translation.
        radius (torch.Tensor):  (B,), the scaling factor for each batch
        eps (float):            Small constant to avoid division by zero.

    Returns:
        torch.Tensor: (B, N, 3), the translated and scaled point clouds.
    """
    # radius is shape (B,)
    # reshape to (B,1,1)
    if radius.dim() == 1:
        radius = radius.view(-1, 1, 1)

    # 1) Translate by subtracting the center
    points_centered = points - center  # (B,N,3) - (B,1,3) -> (B,N,3)

    # 2) Scale by dividing by radius (plus small eps for numerical safety)
    points_sphered = points_centered / (radius + eps)  # broadcast -> (B,N,3)

    return points_sphered

def desphere_points(points_sphered: torch.Tensor,
                    center: torch.Tensor,
                    radius: torch.Tensor,
                    eps: float = 1e-6) -> torch.Tensor:
    """
    Reverses the `sphere_points` transformation. Given sphered points,
    this function recovers their original position/scale before
    centering and radius-scaling were applied.

    Args:
        points_sphered (torch.Tensor): (B, N, 3), batch of previously sphered point clouds.
        center (torch.Tensor):         (B, 1, 3), centers used for the original translation.
        radius (torch.Tensor):         (B,),      radii used for the original scaling.
        eps (float):                   Small constant to match the one used
                                       in `sphere_points` to avoid division by zero.

    Returns:
        torch.Tensor: (B, N, 3), the original point clouds before `sphere_points`.
    """
    # Reshape radius for broadcasting, matching the shape in sphere_points
    if radius.dim() == 1:
        radius = radius.view(-1, 1, 1)

    # 1) Undo the scaling
    points_rescaled = points_sphered * (radius + eps)

    # 2) Undo the translation
    points_original = points_rescaled + center

    return points_original
    

def sphere_pose(R: torch.Tensor,
                T: torch.Tensor,
                centre: torch.Tensor,   # (B, 1, 3)
                radius: torch.Tensor   # (B,)
               ) -> (torch.Tensor, torch.Tensor):
    """
    Given the original rotation & translation (R, T), as well as the point-cloud
    centre and radius used for normalization, compute (R_norm, T_norm)
    in the normalized space.
    
    Args:
        R (torch.Tensor):       (B, 3, 3) rotation in original coords
        T (torch.Tensor):       (B, 3)    translation in original coords
        centre (torch.Tensor):  (B, 1, 3) centre of the source point cloud
        radius (torch.Tensor):  (B,)      scaling factor for each batch

    Returns:
        R_norm (torch.Tensor): (B, 3, 3) rotation in normalized coords
        T_norm (torch.Tensor): (B, 3)    translation in normalized coords
    """
    # -- Rotation is unchanged by uniform scaling
    R_norm = R

    # -- "C" is (B,3)
    C = centre.squeeze(1)  # (B,1,3) -> (B,3)

    # -- We need T_norm = (T + R*C - C) / r
    #    Expand shapes for batch:
    #    R:      (B,3,3)
    #    C:      (B,3)
    #    T:      (B,3)
    #    radius: (B,)
    # 1) R*C via batch-matrix multiply
    RC = torch.bmm(R, C.unsqueeze(-1)).squeeze(-1)  # (B,3)
    # 2) Broadcast radius as (B,1) to divide a (B,3)
    T_norm = (T + RC - C) / radius.unsqueeze(-1)    # (B,3)

    return R_norm, T_norm


def desphere_pose(R_norm: torch.Tensor,
                  T_norm: torch.Tensor,
                  centre: torch.Tensor,  # (B,1,3)
                  radius: torch.Tensor   # (B,)
                 ) -> (torch.Tensor, torch.Tensor):
    """
    Recover the original rotation (R) and translation (T) in unnormalized coords,
    given the normalized (R_norm, T_norm) from `sphere_pose`.
    
    Args:
        R_norm (torch.Tensor):   (B, 3, 3) rotation in normalized coords
        T_norm (torch.Tensor):   (B, 3)    translation in normalized coords
        centre (torch.Tensor):   (B, 1, 3) centre of the source point cloud
        radius (torch.Tensor):   (B,)      scaling factor used to sphere

    Returns:
        R (torch.Tensor): (B, 3, 3) rotation in original coords
        T (torch.Tensor): (B, 3)    translation in original coords
    """
    # -- Rotation is unchanged by uniform scaling
    R = R_norm

    # -- "C" is (B,3)
    C = centre.squeeze(1)  # (B,1,3) -> (B,3)

    # -- We need T = C + r*T_norm - R*C
    RC = torch.bmm(R, C.unsqueeze(-1)).squeeze(-1)     # (B,3)
    T = C + radius.unsqueeze(-1) * T_norm - RC         # (B,3)

    return R, T