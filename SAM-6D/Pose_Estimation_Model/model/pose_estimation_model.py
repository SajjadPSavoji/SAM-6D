import torch
import torch.nn as nn

from feature_extraction import ViTEncoder
from coarse_point_matching import CoarsePointMatching
from fine_point_matching import FinePointMatching
from transformer import GeometricStructureEmbedding
from model_utils import sample_pts_feats
from vis_utils import features_to_colors, visualize_points_3d, visualize_points_3d_two_sets


class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()
        self.cfg = cfg
        self.coarse_npoint = cfg.coarse_npoint
        self.fine_npoint = cfg.fine_npoint

        self.feature_extraction = ViTEncoder(cfg.feature_extraction, self.fine_npoint)
        self.geo_embedding = GeometricStructureEmbedding(cfg.geo_embedding)
        self.coarse_point_matching = CoarsePointMatching(cfg.coarse_point_matching, return_feat=True)
        self.fine_point_matching = FinePointMatching(cfg.fine_point_matching)

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
        

        # # # visualization
        for i in range(end_points['init_R'].size(0)):
            gt_r = end_points['init_R'][i:i+1]
            gt_t = end_points['init_t'][i:i+1]
            gt_pts = (sparse_pm-gt_t.unsqueeze(1))@gt_r
            visualize_points_3d_two_sets(
                gt_pts.squeeze(0).cpu().numpy(),
                sparse_po.squeeze(0).cpu().numpy(), 
                f"sparse_pm_hypo{i}", s=5)
            break
        breakpoint()


        # if self.training:
        #     gt_r = end_points['rotation_label']
        #     gt_t = end_points['translation_label'] / (radius.reshape(-1, 1)+1e-6)
        # else:
        #     gt_r = end_points['init_R']
        #     gt_t = end_points['init_t']

        # gt_pts = (sparse_pm-gt_t.unsqueeze(1))@gt_r
        # visualize_points_3d(gt_pts.squeeze(0).cpu().numpy(), "sparse_pm",c=color_m, s=5)
        # visualize_points_3d(sparse_po.squeeze(0).cpu().numpy(), "sparse_po",c=color_o,s=5)
        # visualize_points_3d(gt_pts.squeeze(0).cpu().numpy(), "sparse_pm_uniform", s=5)
        # visualize_points_3d(sparse_po.squeeze(0).cpu().numpy(), "sparse_po_uniform", s=5)
        # visualize_points_3d(gt_pts.squeeze(0).cpu().numpy(), f"sparse_pm_bg_L_Last",c=bg_scores1.squeeze(0).cpu().detach().numpy(), s=5, cmap="rainbow")
        # visualize_points_3d(sparse_po.squeeze(0).cpu().numpy(), f"sparse_po_bg_L_Last",c=bg_scores2.squeeze(0).cpu().detach().numpy(), s=5, cmap="rainbow")


        # fine_point_matching
        end_points = self.fine_point_matching(
            dense_pm, dense_fm, geo_embedding_m, fps_idx_m,
            dense_po, dense_fo, geo_embedding_o, fps_idx_o,
            radius, end_points
        )

        return end_points

