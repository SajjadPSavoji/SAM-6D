import torch
import torch.nn as nn

from feature_extraction import ViTEncoder
from coarse_point_matching import CoarsePointMatching
from fine_point_matching import FinePointMatching
from transformer import GeometricStructureEmbedding
from model_utils import sample_pts_feats, knn_linear_falloff_interpolate_3d, sample_pts_and_feats_nonuniform

from vis_utils import visualize_points_3d, features_to_colors

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
        all_pm, all_fm, all_po, all_fo, radius = self.feature_extraction(end_points)

        # pre-compute geometric embeddings for geometric transformer
        bg_point = torch.ones(all_pm.size(0),1,3).float().to(all_pm.device) * 100

        sparse_pm, sparse_fm, fps_idx_m = sample_pts_feats(
            all_pm, all_fm, self.coarse_npoint, return_index=True
        )
        geo_embedding_m = self.geo_embedding(torch.cat([bg_point, sparse_pm], dim=1))

        sparse_po, sparse_fo, fps_idx_o = sample_pts_feats(
            all_po, all_fo, self.coarse_npoint, return_index=True
        )
        geo_embedding_o = self.geo_embedding(torch.cat([bg_point, sparse_po], dim=1))

        # coarse_point_matching
        end_points, bg_scores1, bg_scores2, of1, of2 = self.coarse_point_matching(
            sparse_pm, sparse_fm, geo_embedding_m,
            sparse_po, sparse_fo, geo_embedding_o,
            radius, end_points,
        )

        # # # visualization
        # color_m = features_to_colors(of1.squeeze(0).cpu().detach().numpy()[1:, :])
        # color_o = features_to_colors(of2.squeeze(0).cpu().detach().numpy()[1:, :])
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

   
        # interpolate course bg_scores to all points
        k = 196
        # all_scores1 = knn_linear_falloff_interpolate_3d(bg_scores1, sparse_pm, all_pm, k)
        all_scores2 = knn_linear_falloff_interpolate_3d(bg_scores2, sparse_po, all_po, k)
        # sample n_fine poits from all points using inverse of scores so the lower the score, the higher the probability
        dense_po, dense_fo, fps_idx_o = sample_pts_and_feats_nonuniform(all_po, all_fo, all_scores2, fps_idx_o, self.fine_npoint, self.coarse_npoint)
        # dense_pm, dense_fm, fps_idx_m = sample_pts_and_feats_nonuniform(all_pm, all_fm, all_scores1, fps_idx_m, self.fine_npoint, self.coarse_npoint)
        dense_pm, dense_fm, fps_idx_m = all_pm, all_fm, fps_idx_m

        # gather fps points, shape => (B, M, 3), (B, M, F)
        fps_idx_expanded_po = fps_idx_o.unsqueeze(-1).expand(-1, -1, 3)  # [B, M, 3]
        fps_po = torch.gather(all_po, dim=1, index=fps_idx_expanded_po.long())  # [B, M, 3]

        # fps_idx_expanded_pm = fps_idx_m.unsqueeze(-1).expand(-1, -1, 3)  # [B, M, 3]
        # fps_pm = torch.gather(all_pm, dim=1, index=fps_idx_expanded_pm.long())  # [B, M, 3]

        # geo_embedding_m = self.geo_embedding(torch.cat([bg_point, fps_pm], dim=1))
        geo_embedding_o = self.geo_embedding(torch.cat([bg_point, fps_po], dim=1))

        # # visualize bg_scores all po
        # visualize_points_3d(all_po.squeeze(0).cpu().numpy(), f"all_po_bg_int_k{k}",c=all_scores2.squeeze(0).cpu().detach().numpy(), s=1, cmap="rainbow")
        # visualize_points_3d(all_po.squeeze(0).cpu().numpy(), f"all_po_uniform", s=1)
        # visualize_points_3d(dense_po.squeeze(0).cpu().numpy(), "dense_po_non_uniform", s=1)

        # all_gt_pts = (all_pm-gt_t.unsqueeze(1))@gt_r
        # dense_gt_pts = (dense_pm-gt_t.unsqueeze(1))@gt_r
        # visualize_points_3d(all_gt_pts.squeeze(0).cpu().numpy(), f"all_pm_bg_int_k{k}",c=all_scores1.squeeze(0).cpu().detach().numpy(), s=1, cmap="rainbow")
        # visualize_points_3d(all_gt_pts.squeeze(0).cpu().numpy(), f"all_pm_uniform", s=1)
        # visualize_points_3d(dense_gt_pts.squeeze(0).cpu().numpy(), "dense_pm_non_uniform", s=1)

        # breakpoint()
        
        # # also sample uniform
        # dense_uni_po, dense_uni_fo, fps_idx_o_uni = sample_pts_feats(
        #     all_po, all_fo, self.fine_npoint, return_index=True
        # )
        # visualize_points_3d(dense_uni_po.squeeze(0).cpu().numpy(), "dense_po_uniform", s=1)


        # fine_point_matching
        end_points, bg_scores1, bg_scores2, of1, of2 = self.fine_point_matching(
            dense_pm, dense_fm, geo_embedding_m, fps_idx_m,
            dense_po, dense_fo, geo_embedding_o, fps_idx_o,
            radius, end_points
        )

        # color_m = features_to_colors(of1.squeeze(0).cpu().detach().numpy()[1:, :])
        # color_o = features_to_colors(of2.squeeze(0).cpu().detach().numpy()[1:, :])
        # gt_pts = (dense_pm-gt_t.unsqueeze(1))@gt_r
        # print(len(gt_pts.squeeze(0)), len(dense_po.squeeze(0)))
        # visualize_points_3d(gt_pts.squeeze(0).cpu().numpy(), "dense_pm",c=color_m, s=1)
        # visualize_points_3d(dense_po.squeeze(0).cpu().numpy(), "dense_po",c=color_o, s=1)
        # visualize_points_3d(gt_pts.squeeze(0).cpu().numpy(), f"dense_pm_bg_L_Last",c=bg_scores1.squeeze(0).cpu().detach().numpy(), s=1, cmap="rainbow")
        # visualize_points_3d(dense_po.squeeze(0).cpu().numpy(), f"dense_po_bg_L_Last",c=bg_scores2.squeeze(0).cpu().detach().numpy(), s=1, cmap="rainbow")
        # breakpoint()

        return end_points

