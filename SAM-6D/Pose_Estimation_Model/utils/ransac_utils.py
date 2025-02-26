import open3d as o3d
# ignore o3d wqarning logs
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
import torch

def global_registration(source_down, target_down, source_fpfh,target_fpfh, distance_threshold=.075):

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, distance_threshold=0.075):

    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result

def refine_registration(source, target, source_fpfh, target_fpfh, result_ransac, distance_threshold=0.02):

    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

def prep_for_ransace(p, f):
    # preparing format for open3d ransac
    pcd_dsdv = o3d.pipelines.registration.Feature()
    pcd_dsdv.data = f.T
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(p)
    pcd.estimate_normals()
    return pcd, pcd_dsdv

def t_R_from_transformatio(ransac_result):
    t = ransac_result.transformation[:-1, 3]
    R = ransac_result.transformation[:-1, :3]
    return t, R

def add_pose_to_endpoints(end_points, course_pose, fine_pose, radius):
    device = end_points['pts'].device
    init_t, init_R = t_R_from_transformatio(course_pose)
    pred_t, pred_R = t_R_from_transformatio(fine_pose)
    score = fine_pose.fitness
    end_points['init_t'] = torch.tensor(init_t.copy(), device=device).unsqueeze(0)
    end_points['init_R'] = torch.tensor(init_R.copy(), device=device).unsqueeze(0)
    end_points['pred_t'] = torch.tensor(pred_t.copy(), device=device).unsqueeze(0) * (radius.reshape(-1, 1)+1e-6)
    end_points['pred_R'] = torch.tensor(pred_R.copy(), device=device).unsqueeze(0)
    end_points['pred_pose_score'] = torch.tensor([score], device=device)
    return end_points