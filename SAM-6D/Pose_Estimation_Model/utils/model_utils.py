import torch
import torch.nn as nn
from torch.nn import functional as F
import math

import numpy as np

from pointnet2_utils import (
    gather_operation,
    furthest_point_sample,
)

from vis_utils import visualize_points_3d, features_to_colors


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed



def sample_pts_feats(pts, feats, npoint=2048, return_index=False):
    '''
        pts: B*N*3
        feats: B*N*C
    '''
    sample_idx = furthest_point_sample(pts, npoint)
    pts = gather_operation(pts.transpose(1,2).contiguous(), sample_idx)
    pts = pts.transpose(1,2).contiguous()
    feats = gather_operation(feats.transpose(1,2).contiguous(), sample_idx)
    feats = feats.transpose(1,2).contiguous()
    if return_index:
        return pts, feats, sample_idx
    else:
        return pts, feats


def get_chosen_pixel_feats(img, choose):
    shape = img.size()
    if len(shape) == 3:
        pass
    elif len(shape) == 4:
        B, C, H, W = shape
        img = img.reshape(B, C, H*W)
    else:
        assert False

    choose = choose.unsqueeze(1).repeat(1, C, 1)
    x = torch.gather(img, 2, choose).contiguous()
    return x.transpose(1,2).contiguous()


def pairwise_distance(
    x: torch.Tensor, y: torch.Tensor, normalized: bool = False, channel_first: bool = False
) -> torch.Tensor:
    r"""Pairwise distance of two (batched) point clouds.

    Args:
        x (Tensor): (*, N, C) or (*, C, N)
        y (Tensor): (*, M, C) or (*, C, M)
        normalized (bool=False): if the points are normalized, we have "x2 + y2 = 1", so "d2 = 2 - 2xy".
        channel_first (bool=False): if True, the points shape is (*, C, N).

    Returns:
        dist: torch.Tensor (*, N, M)
    """
    if channel_first:
        channel_dim = -2
        xy = torch.matmul(x.transpose(-1, -2), y)  # [(*, C, N) -> (*, N, C)] x (*, C, M)
    else:
        channel_dim = -1
        xy = torch.matmul(x, y.transpose(-1, -2))  # (*, N, C) x [(*, M, C) -> (*, C, M)]
    if normalized:
        sq_distances = 2.0 - 2.0 * xy
    else:
        x2 = torch.sum(x ** 2, dim=channel_dim).unsqueeze(-1)  # (*, N, C) or (*, C, N) -> (*, N) -> (*, N, 1)
        y2 = torch.sum(y ** 2, dim=channel_dim).unsqueeze(-2)  # (*, M, C) or (*, C, M) -> (*, M) -> (*, 1, M)
        sq_distances = x2 - 2 * xy + y2
    sq_distances = sq_distances.clamp(min=0.0)
    return sq_distances


def compute_feature_similarity(feat1, feat2, type='cosine', temp=1.0, normalize_feat=True):
    r'''
    Args:
        feat1 (Tensor): (B, N, C)
        feat2 (Tensor): (B, M, C)

    Returns:
        atten_mat (Tensor): (B, N, M)
    '''
    if normalize_feat:
        feat1 = F.normalize(feat1, p=2, dim=2)
        feat2 = F.normalize(feat2, p=2, dim=2)

    if type == 'cosine':
        atten_mat = feat1 @ feat2.transpose(1,2)
    elif type == 'L2':
        atten_mat = torch.sqrt(pairwise_distance(feat1, feat2))
    else:
        assert False

    atten_mat = atten_mat / temp

    return atten_mat



def aug_pose_noise(gt_r, gt_t,
                std_rots=[15, 10, 5, 1.25, 1],
                max_rot=45,
                sel_std_trans=[0.2, 0.2, 0.2],
                max_trans=0.8):

    B = gt_r.size(0)
    device = gt_r.device

    std_rot = np.random.choice(std_rots)
    angles = torch.normal(mean=0, std=std_rot, size=(B, 3)).to(device=device)
    angles = angles.clamp(min=-max_rot, max=max_rot)
    ones = gt_r.new(B, 1, 1).zero_() + 1
    zeros = gt_r.new(B, 1, 1).zero_()
    a1 = angles[:,0].reshape(B, 1, 1) * np.pi / 180.0
    a1 = torch.cat(
        [torch.cat([torch.cos(a1), -torch.sin(a1), zeros], dim=2),
        torch.cat([torch.sin(a1), torch.cos(a1), zeros], dim=2),
        torch.cat([zeros, zeros, ones], dim=2)], dim=1
    )
    a2 = angles[:,1].reshape(B, 1, 1) * np.pi / 180.0
    a2 = torch.cat(
        [torch.cat([ones, zeros, zeros], dim=2),
        torch.cat([zeros, torch.cos(a2), -torch.sin(a2)], dim=2),
        torch.cat([zeros, torch.sin(a2), torch.cos(a2)], dim=2)], dim=1
    )
    a3 = angles[:,2].reshape(B, 1, 1) * np.pi / 180.0
    a3 = torch.cat(
        [torch.cat([torch.cos(a3), zeros, torch.sin(a3)], dim=2),
        torch.cat([zeros, ones, zeros], dim=2),
        torch.cat([-torch.sin(a3), zeros, torch.cos(a3)], dim=2)], dim=1
    )
    rand_rot = a1 @ a2 @ a3

    rand_trans = torch.normal(
        mean=torch.zeros([B, 3]).to(device),
        std=torch.tensor(sel_std_trans, device=device).view(1, 3),
    )
    rand_trans = torch.clamp(rand_trans, min=-max_trans, max=max_trans)

    rand_rot = gt_r @ rand_rot
    rand_trans = gt_t + rand_trans
    rand_trans[:,2] = torch.clamp(rand_trans[:,2], min=1e-6)

    return rand_rot.detach(), rand_trans.detach()


def compute_coarse_Rt(
    atten,
    pts1,
    pts2,
    model_pts=None,
    n_proposal1=6000,
    n_proposal2=300,
):
    WSVD = WeightedProcrustes()

    B, N1, _ = pts1.size()
    N2 = pts2.size(1)
    device = pts1.device

    if model_pts is None:
        model_pts = pts2
    expand_model_pts = model_pts.unsqueeze(1).repeat(1,n_proposal2,1,1).reshape(B*n_proposal2, -1, 3)

    # compute soft assignment matrix
    pred_score = torch.softmax(atten, dim=2) * torch.softmax(atten, dim=1)
    pred_label1 = torch.max(pred_score[:,1:,:], dim=2)[1]
    pred_label2 = torch.max(pred_score[:,:,1:], dim=1)[1]
    weights1 = (pred_label1>0).float()
    weights2 = (pred_label2>0).float()

    pred_score = pred_score[:, 1:, 1:].contiguous()
    pred_score = pred_score * weights1.unsqueeze(2) * weights2.unsqueeze(1)
    pred_score = pred_score.reshape(B, N1*N2) ** 1.5

    # sample pose hypothese
    cumsum_weights = torch.cumsum(pred_score, dim=1)
    cumsum_weights /= (cumsum_weights[:, -1].unsqueeze(1).contiguous()+1e-8)
    idx = torch.searchsorted(cumsum_weights, torch.rand(B, n_proposal1*3, device=device))
    idx1, idx2 = idx.div(N2, rounding_mode='floor'), idx % N2
    idx1 = torch.clamp(idx1, max=N1-1).unsqueeze(2).repeat(1,1,3)
    idx2 = torch.clamp(idx2, max=N2-1).unsqueeze(2).repeat(1,1,3)

    p1 = torch.gather(pts1, 1, idx1).reshape(B,n_proposal1,3,3).reshape(B*n_proposal1,3,3)
    p2 = torch.gather(pts2, 1, idx2).reshape(B,n_proposal1,3,3).reshape(B*n_proposal1,3,3)
    pred_rs, pred_ts = WSVD(p2, p1, None)
    pred_rs = pred_rs.reshape(B, n_proposal1, 3, 3)
    pred_ts = pred_ts.reshape(B, n_proposal1, 1, 3)

    p1 = p1.reshape(B, n_proposal1, 3, 3)
    p2 = p2.reshape(B, n_proposal1, 3, 3)
    dis = torch.norm((p1 - pred_ts) @ pred_rs - p2, dim=3).mean(2)
    idx = torch.topk(dis, n_proposal2, dim=1, largest=False)[1]
    pred_rs = torch.gather(pred_rs, 1, idx.reshape(B,n_proposal2,1,1).repeat(1,1,3,3))
    pred_ts = torch.gather(pred_ts, 1, idx.reshape(B,n_proposal2,1,1).repeat(1,1,1,3))

    # pose selection
    transformed_pts = (pts1.unsqueeze(1) - pred_ts) @ pred_rs
    transformed_pts = transformed_pts.reshape(B*n_proposal2, -1, 3)
    dis = torch.sqrt(pairwise_distance(transformed_pts, expand_model_pts))
    dis = dis.min(2)[0].reshape(B, n_proposal2, -1)
    scores = weights1.unsqueeze(1).sum(2) / ((dis * weights1.unsqueeze(1)).sum(2) + + 1e-8)
    idx = scores.max(1)[1]
    pred_R = torch.gather(pred_rs, 1, idx.reshape(B,1,1,1).repeat(1,1,3,3)).squeeze(1)
    pred_t = torch.gather(pred_ts, 1, idx.reshape(B,1,1,1).repeat(1,1,1,3)).squeeze(2).squeeze(1)
    return pred_R, pred_t



def compute_fine_Rt(
    atten,
    pts1,
    pts2,
    model_pts=None,
    dis_thres=0.15
):
    if model_pts is None:
        model_pts = pts2

    # compute pose
    WSVD = WeightedProcrustes(weight_thresh=0.0)
    assginment_mat = torch.softmax(atten, dim=2) * torch.softmax(atten, dim=1)
    label1 = torch.max(assginment_mat[:,1:,:], dim=2)[1]
    label2 = torch.max(assginment_mat[:,:,1:], dim=1)[1]

    assginment_mat = assginment_mat[:, 1:, 1:] * (label1>0).float().unsqueeze(2) * (label2>0).float().unsqueeze(1)
    # max_idx = torch.max(assginment_mat, dim=2, keepdim=True)[1]
    # pred_pts = torch.gather(pts2, 1, max_idx.expand_as(pts1))
    normalized_assginment_mat = assginment_mat / (assginment_mat.sum(2, keepdim=True) + 1e-6)
    pred_pts = normalized_assginment_mat @ pts2

    assginment_score = assginment_mat.sum(2)
    pred_R, pred_t = WSVD(pred_pts, pts1, assginment_score)

    # compute score
    pred_pts = (pts1 - pred_t.unsqueeze(1)) @ pred_R
    dis = torch.sqrt(pairwise_distance(pred_pts, model_pts)).min(2)[0]
    mask = (label1>0).float()
    pose_score = (dis < dis_thres).float()
    pose_score = (pose_score * mask).sum(1) / (mask.sum(1) + 1e-8)
    pose_score = pose_score * mask.mean(1)

    return pred_R, pred_t, pose_score



def weighted_procrustes(
    src_points,
    ref_points,
    weights=None,
    weight_thresh=0.0,
    eps=1e-5,
    return_transform=False,
    src_centroid = None,
    ref_centroid = None,
):
    r"""Compute rigid transformation from `src_points` to `ref_points` using weighted SVD.

    Modified from [PointDSC](https://github.com/XuyangBai/PointDSC/blob/master/models/common.py).

    Args:
        src_points: torch.Tensor (B, N, 3) or (N, 3)
        ref_points: torch.Tensor (B, N, 3) or (N, 3)
        weights: torch.Tensor (B, N) or (N,) (default: None)
        weight_thresh: float (default: 0.)
        eps: float (default: 1e-5)
        return_transform: bool (default: False)

    Returns:
        R: torch.Tensor (B, 3, 3) or (3, 3)
        t: torch.Tensor (B, 3) or (3,)
        transform: torch.Tensor (B, 4, 4) or (4, 4)
    """
    if src_points.ndim == 2:
        src_points = src_points.unsqueeze(0)
        ref_points = ref_points.unsqueeze(0)
        if weights is not None:
            weights = weights.unsqueeze(0)
        squeeze_first = True
    else:
        squeeze_first = False

    batch_size = src_points.shape[0]
    if weights is None:
        weights = torch.ones_like(src_points[:, :, 0])
    weights = torch.where(torch.lt(weights, weight_thresh), torch.zeros_like(weights), weights)
    weights = weights / (torch.sum(weights, dim=1, keepdim=True) + eps)
    weights = weights.unsqueeze(2)  # (B, N, 1)

    if src_centroid is None:
        src_centroid = torch.sum(src_points * weights, dim=1, keepdim=True)  # (B, 1, 3)
    elif len(src_centroid.size()) == 2:
        src_centroid = src_centroid.unsqueeze(1)
    src_points_centered = src_points - src_centroid  # (B, N, 3)

    if ref_centroid is None:
        ref_centroid = torch.sum(ref_points * weights, dim=1, keepdim=True)  # (B, 1, 3)
    elif len(ref_centroid.size()) == 2:
        ref_centroid = ref_centroid.unsqueeze(1)
    ref_points_centered = ref_points - ref_centroid  # (B, N, 3)

    H = src_points_centered.permute(0, 2, 1) @ (weights * ref_points_centered)
    U, _, V = torch.svd(H)
    Ut, V = U.transpose(1, 2), V
    eye = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).to(src_points.device)
    eye[:, -1, -1] = torch.sign(torch.det(V @ Ut))
    R = V @ eye @ Ut

    t = ref_centroid.permute(0, 2, 1) - R @ src_centroid.permute(0, 2, 1)
    t = t.squeeze(2)

    if return_transform:
        transform = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
        transform[:, :3, :3] = R
        transform[:, :3, 3] = t
        if squeeze_first:
            transform = transform.squeeze(0)
        return transform
    else:
        if squeeze_first:
            R = R.squeeze(0)
            t = t.squeeze(0)
        return R, t


class WeightedProcrustes(nn.Module):
    def __init__(self, weight_thresh=0.5, eps=1e-5, return_transform=False):
        super(WeightedProcrustes, self).__init__()
        self.weight_thresh = weight_thresh
        self.eps = eps
        self.return_transform = return_transform

    def forward(self, src_points, tgt_points, weights=None,src_centroid = None,ref_centroid = None):
        return weighted_procrustes(
            src_points,
            tgt_points,
            weights=weights,
            weight_thresh=self.weight_thresh,
            eps=self.eps,
            return_transform=self.return_transform,
            src_centroid=src_centroid,
            ref_centroid=ref_centroid
        )



def knn_linear_falloff_interpolate_3d(
    scores: torch.Tensor,         # (B, N_coarse)
    sparse_points: torch.Tensor,  # (B, N_coarse, 3)
    all_points: torch.Tensor,     # (B, N_all, 3)
    K: int = 8
) -> torch.Tensor:
    """
    KNN-based interpolation where each neighbor's contribution
    is a linear falloff based on distance:

        w_i = (D_max - d_i) / sum_j (D_max - d_j)

    d_i = distance to neighbor i among the top-K neighbors,
    D_max = max(d_j) among those K neighbors.

    Args:
        scores:        (B, N_coarse) scores for each sparse point
        sparse_points: (B, N_coarse, 3) coords of the coarse points
        all_points:    (B, N_all, 3) query points to upsample
        K:             number of neighbors to consider
    Returns:
        interpolated_scores: (B, N_all) upsampled scores
    """

    B, N_coarse, _ = sparse_points.shape
    _, N_all, _    = all_points.shape

    # -------------------------------------------------
    # 1) Compute pairwise distances
    #    dist[b, i, j] => distance from all_points[b, i] to sparse_points[b, j]
    #    => shape (B, N_all, N_coarse)
    # -------------------------------------------------
    diff = all_points.unsqueeze(2) - sparse_points.unsqueeze(1)  # (B, N_all, N_coarse, 3)
    dist = diff.norm(dim=-1)                                     # (B, N_all, N_coarse)

    # -------------------------------------------------
    # 2) Top-K nearest neighbors
    #    => shapes: topk_dist, topk_idx = (B, N_all, K)
    # -------------------------------------------------
    topk_dist, topk_idx = torch.topk(dist, k=K, dim=-1, largest=False)

    # -------------------------------------------------
    # 3) Gather the corresponding scores
    #    neighbor_scores => shape (B, N_all, K)
    # -------------------------------------------------
    scores_expanded = scores.unsqueeze(1).expand(B, N_all, N_coarse)  # (B, N_all, N_coarse)
    neighbor_scores = torch.gather(scores_expanded, 2, topk_idx)      # (B, N_all, K)

    # -------------------------------------------------
    # 4) Compute weights = (D_max - d_i) / sum_j(D_max - d_j)
    #    - D_max is the largest distance among the K neighbors
    # -------------------------------------------------
    # shape => (B, N_all, 1)
    max_dist, _ = topk_dist.max(dim=-1, keepdim=True)  

    # w_i = (D_max - dist_i)
    # shape => (B, N_all, K)
    weights = max_dist - topk_dist
    weights = torch.clamp(weights, min=0.0)  # just to be safe

    # sum across the K neighbors => shape (B, N_all, 1)
    denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-10)

    # normalize => shape (B, N_all, K)
    weights = weights / denom

    # -------------------------------------------------
    # 5) Weighted sum => (B, N_all)
    # -------------------------------------------------
    interpolated_scores = (neighbor_scores * weights).sum(dim=-1)

    return interpolated_scores


def sample_pts_and_feats_nonuniform(
    all_po: torch.Tensor,       # (B, N, 3)
    all_fo: torch.Tensor,       # (B, N, F)
    all_scores2: torch.Tensor,  # (B, N) per-point scores (arbitrary range)
    fps_idx_o: torch.Tensor,    # (B, M) indices to concat at the end
    fine_npoint: int,
    coarse_npoint: int,
):
    """
    1) Normalize all_scores2 per batch into [0,1].
    2) Exclude fps_idx_o from normal sampling by setting their (normalized) score to 1.0 => zero prob after (1 - score).
    3) Sample 'fine_npoint - M' points using cumsum + searchsorted (inverse transform sampling).
    4) Concatenate the points/features from fps_idx_o onto the sampled set.
    5) Return the new index tensor (fps_idx_chosen_po) that corresponds to those appended points.
    
    Returns:
        final_po:           (B, fine_npoint, 3)
        final_fo:           (B, fine_npoint, F)
        fps_idx_chosen_po:  (B, M)   indices in [fine_npoint-M, fine_npoint-1]
                                    telling you where those appended points appear
    """
    B, N, _ = all_po.shape
    D = all_fo.shape[-1]
    M = coarse_npoint
    device = all_po.device
    
    # We plan to sample 'fine_npoint - M' points, then append M points => total 'fine_npoint'
    # (Ensure fine_npoint >= M)
    sample_n = fine_npoint - M
    assert sample_n >= 0, f"fine_npoint={fine_npoint} must be >= M={M}"

    # --------------------------------------------------------
    # 1) Normalize 'all_scores2' per batch to [0,1].
    #    (in case they are not already in [0,1])
    # --------------------------------------------------------
    eps_norm = 1e-10
    # shape: (B, 1)
    scores_min = all_scores2.min(dim=1, keepdim=True).values
    scores_max = all_scores2.max(dim=1, keepdim=True).values
    scores_range = (scores_max - scores_min).clamp_min(eps_norm)
    # shape: (B, N)
    scores_norm = (all_scores2 - scores_min) / scores_range
    
    # --------------------------------------------------------
    # 2) Force excluded points to have zero probability by
    #    setting their normalized score = 1.0 => 1 - 1.0 = 0.
    # --------------------------------------------------------
    scores_clone = scores_norm.clone()
    scores_clone.scatter_(1, fps_idx_o.long(), 1.0)  # set excluded indices to 1
    
    
    # --------------------------------------------------------
    # 3) Build probability distribution via p = (1 - score)
    #    and sample 'sample_n' points with inverse transform
    #    sampling (cumsum + searchsorted).
    # --------------------------------------------------------
    eps = 1e-10
    alpha = 2
    prob = (1.0 - scores_clone).clamp_min(eps)  # shape (B, N)
    prob = prob**alpha 
    cumsum_prob = torch.cumsum(prob, dim=1)    # shape (B, N)
    cumsum_prob /= (cumsum_prob[:, -1].unsqueeze(1).contiguous()+eps)

    # random values ~ U[0,1], shape (B, sample_n)
    rand_vals = torch.rand(B, sample_n, device=device)
    
    # indices from [0..N-1], shape (B, sample_n)
    idx = torch.searchsorted(cumsum_prob, rand_vals)

    # Gather chosen points/features
    idx_expanded_po = idx.unsqueeze(-1).expand(-1, -1, 3)  # [B, sample_n, 3]
    idx_expanded_fo = idx.unsqueeze(-1).expand(-1, -1, D)  # [B, sample_n, F]
    chosen_po = torch.gather(all_po, dim=1, index=idx_expanded_po)  # [B, sample_n, 3]
    chosen_fo = torch.gather(all_fo, dim=1, index=idx_expanded_fo)  # [B, sample_n, F]
    
    # --------------------------------------------------------
    # 4) Concatenate the 'fps_idx_o' points/features at the end
    # --------------------------------------------------------
    # gather fps points, shape => (B, M, 3), (B, M, F)
    fps_idx_expanded_po = fps_idx_o.unsqueeze(-1).expand(-1, -1, 3)  # [B, M, 3]
    fps_idx_expanded_fo = fps_idx_o.unsqueeze(-1).expand(-1, -1, D)  # [B, M, F]
    fps_po = torch.gather(all_po, dim=1, index=fps_idx_expanded_po.long())  # [B, M, 3]
    fps_fo = torch.gather(all_fo, dim=1, index=fps_idx_expanded_fo.long())  # [B, M, F]
    
    # final shapes => (B, sample_n + M, 3/ F) = (B, fine_npoint, 3/ F)
    final_po = torch.cat([chosen_po, fps_po], dim=1)
    final_fo = torch.cat([chosen_fo, fps_fo], dim=1)
    
    # --------------------------------------------------------
    # 5) Build a new index tensor that says:
    #    "The appended M points now live at indices
    #     [sample_n .. sample_n + M - 1] in each batch."
    # --------------------------------------------------------
    fps_idx_chosen_po = torch.arange(
        sample_n, sample_n + M, device=device
    ).unsqueeze(0).expand(B, -1)  # (B, M)

    return final_po, final_fo, fps_idx_chosen_po.int()