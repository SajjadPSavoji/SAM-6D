import torch
import torch.nn as nn

from feature_extraction import ViTEncoder
from coarse_point_matching import CoarsePointMatching
from fine_point_matching import FinePointMatching
from transformer import GeometricStructureEmbedding
from model_utils import sample_pts_feats
from ransac_utils import prep_for_ransace, global_registration, refine_registration,add_pose_to_endpoints
from gedi.gedi_utils import gedi_features


class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()
        self.cfg = cfg
        self.coarse_npoint = cfg.coarse_npoint
        self.fine_npoint = cfg.fine_npoint

        self.feature_extraction = ViTEncoder(cfg.feature_extraction, self.fine_npoint)
        self.geo_embedding = GeometricStructureEmbedding(cfg.geo_embedding)
        self.coarse_point_matching = CoarsePointMatching(cfg.coarse_point_matching)
        self.fine_point_matching = FinePointMatching(cfg.fine_point_matching)

    def forward(self, end_points):
        dense_pm, dense_fm, dense_po, dense_fo, radius = self.feature_extraction(end_points)

        # only support single instance inference right now
        assert dense_pm.size()[0] == 1

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

        # extract features with gedi
        sparse_go, dense_go = gedi_features(sparse_po, dense_po, fps_idx_o)
        sparse_gm, dense_gm = gedi_features(sparse_pm, dense_pm, fps_idx_m)

    
        # global registeration with sparse points and sparse features
        pcd_pm, pcd_fm = prep_for_ransace(sparse_pm, sparse_gm)
        pcd_po, pcd_fo = prep_for_ransace(sparse_po, sparse_go)
        course_pose = global_registration(pcd_po, pcd_pm, pcd_fo, pcd_fm)

        # ICP with course points and course features
        pcd_pm, pcd_fm = prep_for_ransace(dense_pm, dense_gm)
        pcd_po, pcd_fo = prep_for_ransace(dense_po, dense_go)
        fine_pose = refine_registration(pcd_po, pcd_pm, pcd_fo, pcd_fm, course_pose)


        end_points = add_pose_to_endpoints(end_points, course_pose, fine_pose, radius)
    
        return end_points

