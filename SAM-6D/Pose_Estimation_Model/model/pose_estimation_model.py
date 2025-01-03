import torch
import torch.nn as nn

from feature_extraction import ViTEncoder
from coarse_point_matching import CoarsePointMatching
from fine_point_matching import FinePointMatching
from transformer import GeometricStructureEmbedding
from model_utils import sample_pts_feats

from training_dataset import visualize_points_3d, features_to_colors

class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()
        self.cfg = cfg
        self.coarse_npoint = cfg.coarse_npoint
        self.fine_npoint = cfg.fine_npoint

        self.feature_extraction = ViTEncoder(cfg.feature_extraction, self.fine_npoint)
        self.geo_embedding = GeometricStructureEmbedding(cfg.geo_embedding)
        self.coarse_point_matching = CoarsePointMatching(cfg.coarse_point_matching, return_feat=True)
        self.fine_point_matching = FinePointMatching(cfg.fine_point_matching, return_feat=True)

    def forward(self, end_points):
        dense_pm, dense_fm, dense_po, dense_fo, radius = self.feature_extraction(end_points)

        # pre-compute geometric embeddings for geometric transformer
        bg_point = torch.ones(dense_pm.size(0),1,3).float().to(dense_pm.device) * 100

        sparse_pm, sparse_fm, fps_idx_m = sample_pts_feats(
            dense_pm, dense_fm, self.coarse_npoint, return_index=True
        )
        geo_embedding_m = self.geo_embedding(torch.cat([bg_point, sparse_pm], dim=1))

        sparse_po, sparse_fo, fps_idx_o = sample_pts_feats(
            dense_po, dense_fo, self.coarse_npoint, return_index=True
        )
        geo_embedding_o = self.geo_embedding(torch.cat([bg_point, sparse_po], dim=1))

        # coarse_point_matching
        end_points, of1, of2 = self.coarse_point_matching(
            sparse_pm, sparse_fm, geo_embedding_m,
            sparse_po, sparse_fo, geo_embedding_o,
            radius, end_points,
        )

        color_m = features_to_colors(of1.squeeze(0).cpu().detach().numpy()[1:, :])
        color_o = features_to_colors(of2.squeeze(0).cpu().detach().numpy()[1:, :])
        gt_r = end_points['rotation_label']
        gt_t = end_points['translation_label'] / (radius.reshape(-1, 1)+1e-6)
        gt_pts = (sparse_pm-gt_t.unsqueeze(1))@gt_r
        print(len(gt_pts.squeeze(0)), len(sparse_po.squeeze(0)))
        visualize_points_3d(gt_pts.squeeze(0).cpu().numpy(), "sparse_pm",color=color_m, s=5)
        visualize_points_3d(sparse_po.squeeze(0).cpu().numpy(), "sparse_po",color=color_o,s=5)

        # fine_point_matching
        end_points, of1, of2 = self.fine_point_matching(
            dense_pm, dense_fm, geo_embedding_m, fps_idx_m,
            dense_po, dense_fo, geo_embedding_o, fps_idx_o,
            radius, end_points
        )

        color_m = features_to_colors(of1.squeeze(0).cpu().detach().numpy()[1:, :])
        color_o = features_to_colors(of2.squeeze(0).cpu().detach().numpy()[1:, :])
        gt_pts = (dense_pm-gt_t.unsqueeze(1))@gt_r
        print(len(gt_pts.squeeze(0)), len(dense_po.squeeze(0)))
        visualize_points_3d(gt_pts.squeeze(0).cpu().numpy(), "dense_pm",color=color_m)
        visualize_points_3d(dense_po.squeeze(0).cpu().numpy(), "dense_po",color=color_o)

        breakpoint()


        return end_points

