import torch
import numpy as np
import open3d as o3d
from .gedi import GeDi
from pointnet2_utils import gather_operation

def gedi_features(sparse_p, dense_p, fps_idx):

    config = {'dim': 32,                                          # descriptor output dimension
            'samples_per_batch': 500,                             # batches to process the data on GPU
            'samples_per_patch_lrf': 4000,                        # num. of point to process with LRF
            'samples_per_patch_out': 512,                         # num. of points to sample for pointnet++
            'r_lrf': .5,                                          # LRF radius
            'fchkpt_gedi_net': './model/gedi/data/chkpts/3dmatch/chkpt.tar'}   # path to checkpoint

    gedi = GeDi(config=config)

    dense_f = gedi.compute(pts=dense_p[0].detach().cpu(), pcd=dense_p[0].detach().cpu())
    dense_f = torch.tensor(dense_f, device=sparse_p.device).unsqueeze(0).float()
    sparse_f = gather_operation(dense_f.transpose(1,2).contiguous(), fps_idx).transpose(1,2).contiguous()

    return sparse_f, dense_f