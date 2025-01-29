import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np

from pointnet2_utils import (
    gather_operation,
    furthest_point_sample,
)


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


def get_knn_pts_fts(
    pts: torch.Tensor,
    dense_pts: torch.Tensor,
    dense_fts: torch.Tensor,
    k: int = 10
):
    """
    For each 3D point in 'pts', retrieve the k nearest neighbors from 'dense_pts'
    and, if provided, also gather the corresponding features from 'dense_fts'.

    Args:
        pts (torch.Tensor): [B, n_hypo, 3, 3].
            - B = batch size
            - n_hypo = number of hypotheses
            - 3 points per hypothesis
            - each point is 3D
        dense_pts (torch.Tensor): [B, L, 3]
            - B = batch size
            - L = number of dense points
            - each point is 3D
        dense_fts (torch.Tensor, optional): [B, L, D]
            - B = batch size
            - L = number of dense points
            - D = feature dimension per point
            - If None, we skip feature gathering
        k (int): number of nearest neighbors to retrieve

    Returns:
        neighbors (torch.Tensor): [B, n_hypo, 3, k, 3]
            - k nearest neighbor coordinates for each point
        neighbors_fts (torch.Tensor or None): [B, n_hypo, 3, k, D]
            - corresponding features for the k neighbors (if dense_fts is not None),
              otherwise None
    """
    device = pts.device
    B, n_hypo, _, _ = pts.shape  # pts is [B, n_hypo, 3, 3]
    
    # 1) Flatten pts so each row is (x, y, z) for distance computation
    #    pts_flat => [B, n_hypo * 3, 3]
    pts_flat = pts.view(B, n_hypo * 3, 3)

    # 2) Compute pairwise distances: [B, (n_hypo * 3), L]
    dists = torch.cdist(pts_flat, dense_pts, p=2)

    # 3) k-NN indices (smallest distances): [B, (n_hypo * 3), k]
    _, knn_indices = torch.topk(dists, k, dim=2, largest=False)

    # 4) Advanced indexing to retrieve neighbor coordinates:
    #    Build a batch index [B, (n_hypo * 3), k].
    B_idx = torch.arange(B, device=device).view(-1, 1, 1).expand(-1, n_hypo * 3, k)

    # neighbors => [B, (n_hypo * 3), k, 3]
    neighbors = dense_pts[B_idx, knn_indices, :]

    # Reshape to [B, n_hypo, 3, k, 3]
    neighbors = neighbors.view(B, n_hypo, 3, k, 3)

    # 5) If features are provided, gather them as well
    # dense_fts => [B, L, D]
    # neighbors_fts => [B, (n_hypo * 3), k, D]
    neighbors_fts = dense_fts[B_idx, knn_indices, :]

    # Reshape => [B, n_hypo, 3, k, D]
    D = dense_fts.shape[-1]
    neighbors_fts = neighbors_fts.view(B, n_hypo, 3, k, D)

    return neighbors, neighbors_fts


def compute_feature_similarity_5d(
    feat1: torch.Tensor,
    feat2: torch.Tensor,
    similarity_type: str = 'cosine',
    temp: float = 1.0,
    normalize_feat: bool = True
) -> torch.Tensor:
    r"""
    Compute similarity between two 5D feature tensors:
    - feat1, feat2: [B, n_hypo, 3, k, D]

    Returns:
        sim_mat: [B, n_hypo, 3, k, k] similarity matrix
    """

    B, n_hypo, _, k, D = feat1.shape
    assert feat2.shape == (B, n_hypo, 3, k, D), \
        "feat2 must match feat1's shape [B, n_hypo, 3, k, D]."

    # 1) Optionally L2-normalize feats along the last dim
    if normalize_feat:
        feat1 = F.normalize(feat1, p=2, dim=-1)  # along D
        feat2 = F.normalize(feat2, p=2, dim=-1)

    # 2) Reshape to 3D for simpler batch matmul
    #    We combine [B, n_hypo, 3] into a single dimension => B' = B * n_hypo * 3
    Bp = B * n_hypo * 3  # "flattened" batch
    feat1_3d = feat1.view(Bp, k, D)  # [B', k, D]
    feat2_3d = feat2.view(Bp, k, D)  # [B', k, D]

    if similarity_type == 'cosine':
        # Dot product => [B', k, k]
        sim_3d = torch.bmm(feat1_3d, feat2_3d.transpose(1, 2))
    elif similarity_type == 'L2':
        # Example: negative L2 distance or some measure
        # (But typically you'd do pairwise_distance here.)
        # We'll show a placeholder for demonstration.
        # If you truly want L2, you might do something like:
        #   dist_3d = torch.cdist(feat1_3d, feat2_3d, p=2)
        #   sim_3d = -dist_3d  # or some function
        dist_3d = torch.cdist(feat1_3d, feat2_3d, p=2)
        sim_3d = -dist_3d  # turning distance into "similarity"
    else:
        raise ValueError(f"Unknown similarity type: {similarity_type}")

    # 3) Reshape back to [B, n_hypo, 3, k, k]
    sim_mat = sim_3d.view(B, n_hypo, 3, k, k)

    # 4) Apply temperature scaling
    sim_mat = sim_mat / temp

    return sim_mat

import torch

def sinkhorn(
    score_mat: torch.Tensor,
    num_iterations: int = 10
) -> torch.Tensor:
    r"""
    1) Perform log-domain Sinkhorn on a 5D input of shape [B, n_hypo, 3, k, k].
    2) Exponentiate to get a normal (k x k) plan for each cluster.
    3) Embed these three (k x k) blocks into a [B, n_hypo, 3k, 3k] block-diagonal
       matrix with zeros off-diagonal, *without* looping over the cluster dimension.

    Args:
        score_mat (torch.Tensor): [B, n_hypo, 3, k, k]
          e.g., a similarity or negative distance matrix
        num_iterations (int): number of Sinkhorn iterations

    Returns:
        transport_4d (torch.Tensor): [B, n_hypo, 3k, 3k]
            Block-diagonal matrix with each (k x k) sub-block on the diagonal
            and zeros elsewhere (no cluster cross-match).
    """
    device = score_mat.device
    dtype = score_mat.dtype

    B, n_hypo, n_clusters, k, k2 = score_mat.shape
    assert n_clusters == 3, "This example assumes exactly 3 clusters."
    assert k == k2, "score_mat's last two dims must be (k, k)."

    # ---------------------------------------------------------------------
    # 1) Flatten -> [B', k, k], where B' = B*n_hypo*n_clusters
    # ---------------------------------------------------------------------
    Bp = B * n_hypo * n_clusters
    score_mat_3d = score_mat.view(Bp, k, k)  # shape [B', k, k]

    # ---------------------------------------------------------------------
    # 2) Sinkhorn in log domain
    # ---------------------------------------------------------------------
    u = torch.zeros(Bp, k, device=device, dtype=dtype)
    v = torch.zeros(Bp, k, device=device, dtype=dtype)

    for _ in range(num_iterations):
        row_logsum = torch.logsumexp(score_mat_3d + v.unsqueeze(1), dim=2)  # [B', k]
        u = -row_logsum
        col_logsum = torch.logsumexp(score_mat_3d + u.unsqueeze(2), dim=1)  # [B', k]
        v = -col_logsum

    log_transport_3d = score_mat_3d + u.unsqueeze(2) + v.unsqueeze(1)

    # Reshape back to [B, n_hypo, 3, k, k] and exponentiate
    log_transport_5d = log_transport_3d.view(B, n_hypo, n_clusters, k, k)
    transport_5d = torch.exp(log_transport_5d)  # shape => [B, n_hypo, 3, k, k]

    # ---------------------------------------------------------------------
    # 3) Build the final [B, n_hypo, 3k, 3k] block-diagonal matrix WITHOUT looping
    # ---------------------------------------------------------------------
    # Allocate output
    transport_4d = torch.zeros(
        (B, n_hypo, n_clusters*k, n_clusters*k),
        device=device,
        dtype=dtype
    )

    # We want:
    #   transport_4d[b, n, i*k + row, i*k + col] = transport_5d[b, n, i, row, col]
    #
    # We'll do this in a single advanced-indexing assignment.

    # 3.1) Build the per-cluster row/col offsets
    cluster_idx = torch.arange(n_clusters, device=device).view(-1, 1, 1)  # [3,1,1]
    row_idx     = torch.arange(k, device=device).view(1, -1, 1)          # [1,k,1]
    col_idx     = torch.arange(k, device=device).view(1, 1, -1)          # [1,1,k]

    # final_row, final_col => [3, k, k]  with block offsets
    final_row = cluster_idx * k + row_idx  # shape [3,k,k]
    final_col = cluster_idx * k + col_idx  # shape [3,k,k]

    # 3.2) We also need batch/hypo indexing to broadcast
    # For the left side: transport_4d[:, :, final_row, final_col]
    # For the right side: transport_5d[:, :, cluster_idx, row_idx, col_idx]
    B_idx = torch.arange(B, device=device).view(-1, 1, 1, 1, 1)            # [B,1,1,1,1]
    N_idx = torch.arange(n_hypo, device=device).view(1, -1, 1, 1, 1)       # [1,n_hypo,1,1,1]

    # Expand final_row/final_col to [1,1,3,k,k], cluster_idx/row_idx/col_idx likewise
    final_row = final_row.unsqueeze(0).unsqueeze(0)  # => [1,1,3,k,k]
    final_col = final_col.unsqueeze(0).unsqueeze(0)
    cluster_idx = cluster_idx.unsqueeze(0).unsqueeze(0)  # => [1,1,3,1,1]
    row_idx     = row_idx.unsqueeze(0).unsqueeze(0)      # => [1,1,3,k,1]
    col_idx     = col_idx.unsqueeze(0).unsqueeze(0)      # => [1,1,3,1,k]

    # 3.3) Advanced indexing (both sides match shape [B, n_hypo, 3, k, k])
    # Left side => transport_4d[B_idx, N_idx, final_row, final_col]
    # Right side => transport_5d[B_idx, N_idx, cluster_idx, row_idx, col_idx]
    transport_4d[B_idx, N_idx, final_row, final_col] = transport_5d[
        B_idx,
        N_idx,
        cluster_idx,
        row_idx,
        col_idx
    ]

    return transport_4d



def compute_coarse_Rt(
    atten,
    pts1,
    pts2,
    model_pts=None,
    n_proposal1=6000,
    n_proposal2=300,
    dense_pts1=None,
    dense_fts1=None,
    dense_pts2=None,
    dense_fts2=None
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
   
    p1 = torch.gather(pts1, 1, idx1).reshape(B,n_proposal1,3,3)
    p2 = torch.gather(pts2, 1, idx2).reshape(B,n_proposal1,3,3)

    p1_neighbors, p1_neighbors_fts = get_knn_pts_fts(p1, dense_pts1, dense_fts1) #[B, n_hypo, 3, k, 3], [B, n_hypo, 3, k, D]
    p2_neighbors, p2_neighbors_fts = get_knn_pts_fts(p2, dense_pts2, dense_fts2)
    scores_p1_p2_neighbors = compute_feature_similarity_5d(p1_neighbors_fts, p2_neighbors_fts) #[B, n_hypo, 3, k, k]
    assignment_p1_p2_neighbors = sinkhorn(scores_p1_p2_neighbors) #[B, n_hypo, 3k, 3k]
    p1_neighbors = p1_neighbors.reshape(B, n_proposal1, -1, 3) #[B, n_hypo, 3k, 3]
    p2_neighbors = p2_neighbors.reshape(B, n_proposal1, -1, 3) #[B, n_hypo, 3k, 3]
    row_sums = assignment_p1_p2_neighbors.sum(dim=3, keepdim=True) + 1e-6     #    shape => [B, n_hypo, 3k, 1]
    normalized_assign = assignment_p1_p2_neighbors / row_sums     #    shape => [B, n_hypo, 3k, 3k]
    B, n_hypo, three_k, _ = assignment_p1_p2_neighbors.shape
    normalized_assign_4d = normalized_assign.view(B * n_hypo, three_k, three_k) # [B*n_hypo, 3k, 3k]
    p2_neighbors_4d = p2_neighbors.view(B * n_hypo, three_k, 3) # [B*n_hypo, 3k, 3]
    pred_p2_soft_4d = torch.bmm(normalized_assign_4d, p2_neighbors_4d) # [B*n_hypo, 3k, 3]
    assignment_score = assignment_p1_p2_neighbors.sum(dim=3).view(B * n_hypo, three_k)  # [B, n_hypo, 3k]
    pred_p1_hard_4d = p1_neighbors.view(B * n_hypo, three_k, 3)

    pred_rs, pred_ts = WSVD(pred_p2_soft_4d, pred_p1_hard_4d, assignment_score)
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

