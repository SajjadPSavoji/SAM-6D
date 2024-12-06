
import torch
import torch.nn as nn

from model_utils import pairwise_distance

def compute_correspondence_loss(
    end_points,
    atten_list,
    pts1,
    pts2,
    gt_r,
    gt_t,
    dis_thres=0.15,
    loss_str='coarse'
):
    CE = nn.CrossEntropyLoss(reduction ='none')

    clutter  = end_points['clutter']
    occluded = end_points['occluded']

    # sample occluded labels based on dense indices first
    indices = end_points['fpd_idx_o']
    B, M = indices.size()
    batch_indices = torch.arange(B).unsqueeze(1).expand(B, M)
    occluded = occluded[batch_indices, indices]

    # Sample clutter and occluded labels for course matching
    if loss_str == 'coarse':
        # sample clutter
        indices = end_points['fps_idx_m']
        B, M = indices.size()
        batch_indices = torch.arange(B).unsqueeze(1).expand(B, M)
        clutter = clutter[batch_indices, indices]

        # sample occluded
        indices = end_points['fps_idx_o']
        B, M = indices.size()
        batch_indices = torch.arange(B).unsqueeze(1).expand(B, M)
        occluded = occluded[batch_indices, indices]

    gt_pts = (pts1-gt_t.unsqueeze(1))@gt_r
    dis_mat = torch.sqrt(pairwise_distance(gt_pts, pts2))

    dis1, label1 = dis_mat.min(2)
    # get fg label from clutter
    fg_label1 = 1 - clutter
    label1 = (fg_label1 * (label1.float()+1.0)).long()

    dis2, label2 = dis_mat.min(1)
    # get fg label from occluded
    fg_label2 = 1 - occluded
    label2 = (fg_label2 * (label2.float()+1.0)).long()

    # loss
    for idx, atten in enumerate(atten_list):
        l1 = CE(atten.transpose(1,2)[:,:,1:].contiguous(), label1).mean(1)
        l2 = CE(atten[:,:,1:].contiguous(), label2).mean(1)
        end_points[loss_str + '_loss' + str(idx)] = 0.5 * (l1 + l2)

    # acc
    pred_label = torch.max(atten_list[-1][:,1:,:], dim=2)[1]
    end_points[loss_str + '_acc'] = (pred_label==label1).float().mean(1)

    # pred foreground num
    fg_mask = (pred_label > 0).float()
    end_points[loss_str + '_fg_num'] = fg_mask.sum(1)

    # foreground point dis
    fg_label = fg_mask * (pred_label - 1)
    fg_label = fg_label.long()
    pred_pts = torch.gather(pts2, 1, fg_label.unsqueeze(2).repeat(1,1,3))
    pred_dis = torch.norm(pred_pts-gt_pts, dim=2)
    pred_dis = (pred_dis * fg_mask).sum(1) / (fg_mask.sum(1)+1e-8)
    end_points[loss_str + '_dis'] = pred_dis

    return end_points



class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, end_points):
        out_dicts = {'loss': 0}
        for key in end_points.keys():
            if 'coarse_' in key or 'fine_' in key:
                out_dicts[key] = end_points[key].mean()
                if 'loss' in key:
                    out_dicts['loss'] = out_dicts['loss'] + end_points[key]
        out_dicts['loss'] = torch.clamp(out_dicts['loss'], max=100.0).mean()
        return out_dicts

