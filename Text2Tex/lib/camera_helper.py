import torch

import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from pytorch3d.renderer import (
    PerspectiveCameras,
    look_at_view_transform,
    FoVPerspectiveCameras 
)

# customized
import sys
sys.path.append(".")

from lib.constants import VIEWPOINTS

# ---------------- UTILS ----------------------

def degree_to_radian(d):
    return d * np.pi / 180

def radian_to_degree(r):
    return 180 * r / np.pi

def xyz_to_polar(xyz):
    """ assume y-axis is the up axis """
    
    x, y, z = xyz
    
    theta = 180 * np.arccos(z) / np.pi
    phi = 180 * np.arccos(y) / np.pi

    return theta, phi

def polar_to_xyz(theta, phi, dist):
    """ assume y-axis is the up axis """

    theta = degree_to_radian(theta)
    phi = degree_to_radian(phi)

    x = np.sin(phi) * np.sin(theta) * dist
    y = np.cos(phi) * dist
    z = np.sin(phi) * np.cos(theta) * dist

    return [x, y, z]


# ---------------- VIEWPOINTS ----------------------


def filter_viewpoints(pre_viewpoints: dict, viewpoints: dict):
    """ return the binary mask of viewpoints to be filtered """

    filter_mask = [0 for _ in viewpoints.keys()]
    for i, v in viewpoints.items():
        x_v, y_v, z_v = polar_to_xyz(v["azim"], 90 - v["elev"], v["dist"])

        for _, pv in pre_viewpoints.items():
            x_pv, y_pv, z_pv = polar_to_xyz(pv["azim"], 90 - pv["elev"], pv["dist"])
            sim = cosine_similarity(
                np.array([[x_v, y_v, z_v]]),
                np.array([[x_pv, y_pv, z_pv]])
            )[0, 0]

            if sim > 0.9:
                filter_mask[i] = 1

    return filter_mask


def init_viewpoints(mode, sample_space, init_dist, init_elev, principle_directions, 
    use_principle=True, use_shapenet=False, use_objaverse=False):

    if mode == "predefined":

        (
            dist_list, 
            elev_list, 
            azim_list, 
            sector_list
        ) = init_predefined_viewpoints(sample_space, init_dist, init_elev)

    elif mode == "hemisphere":

        (
            dist_list, 
            elev_list, 
            azim_list, 
            sector_list
        ) = init_hemisphere_viewpoints(sample_space, init_dist)

    else:
        raise NotImplementedError()

    # punishments for views -> in case always selecting the same view
    view_punishments = [1 for _ in range(len(dist_list))]

    if use_principle:

        (
            dist_list, 
            elev_list, 
            azim_list, 
            sector_list,
            view_punishments
        ) = init_principle_viewpoints(
            principle_directions, 
            dist_list, 
            elev_list, 
            azim_list, 
            sector_list,
            view_punishments,
            use_shapenet,
            use_objaverse
        )

    return dist_list, elev_list, azim_list, sector_list, view_punishments


def init_principle_viewpoints(
    principle_directions, 
    dist_list, 
    elev_list, 
    azim_list, 
    sector_list,
    view_punishments,
    use_shapenet=False,
    use_objaverse=False
):

    if use_shapenet:
        key = "shapenet"

        pre_elev_list = [v for v in VIEWPOINTS[key]["elev"]]
        pre_azim_list = [v for v in VIEWPOINTS[key]["azim"]]
        pre_sector_list = [v for v in VIEWPOINTS[key]["sector"]]

        num_principle = 10
        pre_dist_list = [dist_list[0] for _ in range(num_principle)]
        pre_view_punishments = [0 for _ in range(num_principle)]

    elif use_objaverse:
        key = "objaverse"

        pre_elev_list = [v for v in VIEWPOINTS[key]["elev"]]
        pre_azim_list = [v for v in VIEWPOINTS[key]["azim"]]
        pre_sector_list = [v for v in VIEWPOINTS[key]["sector"]]

        num_principle = 10
        pre_dist_list = [dist_list[0] for _ in range(num_principle)]
        pre_view_punishments = [0 for _ in range(num_principle)]
    else:
        num_principle = 6
        pre_elev_list = [v for v in VIEWPOINTS[num_principle]["elev"]]
        pre_azim_list = [v for v in VIEWPOINTS[num_principle]["azim"]]
        pre_sector_list = [v for v in VIEWPOINTS[num_principle]["sector"]]
        pre_dist_list = [dist_list[0] for _ in range(num_principle)]
        pre_view_punishments = [0 for _ in range(num_principle)]

    dist_list = pre_dist_list + dist_list
    elev_list = pre_elev_list + elev_list
    azim_list = pre_azim_list + azim_list
    sector_list = pre_sector_list + sector_list
    view_punishments = pre_view_punishments + view_punishments

    return dist_list, elev_list, azim_list, sector_list, view_punishments


def init_predefined_viewpoints(sample_space, init_dist, init_elev):
    
    viewpoints = VIEWPOINTS[sample_space]

    assert sample_space == len(viewpoints["sector"])

    dist_list = [init_dist for _ in range(sample_space)] # always the same dist
    elev_list = [viewpoints["elev"][i] for i in range(sample_space)]
    azim_list = [viewpoints["azim"][i] for i in range(sample_space)]
    sector_list = [viewpoints["sector"][i] for i in range(sample_space)]

    return dist_list, elev_list, azim_list, sector_list


def init_hemisphere_viewpoints(sample_space, init_dist):
    """
        y is up-axis
    """

    num_points = 2 * sample_space
    ga = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

    flags = []
    elev_list = [] # degree
    azim_list = [] # degree

    for i in range(num_points):
        y = 1 - (i / float(num_points - 1)) * 2  # y goes from 1 to -1

        # only take the north hemisphere
        if y >= 0: 
            flags.append(True)
        else:
            flags.append(False)

        theta = ga * i  # golden angle increment

        elev_list.append(radian_to_degree(np.arcsin(y)))
        azim_list.append(radian_to_degree(theta))

        radius = np.sqrt(1 - y * y)  # radius at y
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

    elev_list = [elev_list[i] for i in range(len(elev_list)) if flags[i]]
    azim_list = [azim_list[i] for i in range(len(azim_list)) if flags[i]]

    dist_list = [init_dist for _ in elev_list]
    sector_list = ["good" for _ in elev_list] # HACK don't define sector names for now

    return dist_list, elev_list, azim_list, sector_list


# ---------------- CAMERAS ----------------------
from typing import Tuple

def init_camera(dist, elev, azim, image_size, device,B):
    # R, T = look_at_view_transform(dist, elev, azim)
    
    # R, T = B2P(B)
    # R = torch.tensor(R).unsqueeze(0)
    # T = torch.tensor(T).unsqueeze(0)

    # R[:, 1] *= -1  # flip y axis
    # T = T @ R  # Make rotation local
    # # # 相机水平视场角度
    # # camera_angle_x = np.rad2deg(1.6352901458740234)  # 以度为单位
    # camera_angle_x = 180
    # # 将水平视场角度转换为 FOV
    # # 计算公式：fov = 2 * arctan(sensor_width / (2 * focal_length))
    # fov = 2.0 * torch.tan(torch.deg2rad(torch.tensor(camera_angle_x)) / 2.0)

    # cameras = FoVPerspectiveCameras(
    #     R=R,
    #     T=T,
    #     device=device,
    #     fov=fov
    # )

    
    R, T = B2P(B)
    R = torch.tensor(R).unsqueeze(0)
    T = _tensor = torch.tensor(T).unsqueeze(0)  #(3,) 转换为 PyTorch 张量，并添加一个维度
    # B = P2B(R.cpu().numpy()[0],T.cpu().numpy()[0])
    image_size = torch.tensor([image_size, image_size]).unsqueeze(0)
    cameras = PerspectiveCameras(R=R, T=T, device=device, image_size=image_size)







    return cameras

def P2B(R:np.ndarray, T:np.ndarray)->np.ndarray:
    P2B_R1 = np.array([[1,0,0],[0,0,-1],[0,1,0]], dtype=np.float64)
    P2B_R2 = np.array([[-1,0,0],[0,1,0],[0,0,-1]], dtype=np.float64)
    P2B_T  = np.array([[-1,0,0],[0,0,1],[0,-1,0]], dtype=np.float64)
    vec4w  = np.array([[0,0,0,1]], dtype=np.float64)
    Bcol3 = P2B_T @ R @ T
    B3x3  = P2B_R1 @ R @ P2B_R2
    B3x4 = np.concatenate([B3x3, Bcol3[:,None]], axis=1)
    B = np.concatenate([B3x4,vec4w], axis=0)
    return B

# blender to pytorch3d
def B2P(B:np.ndarray)->Tuple[np.ndarray, np.ndarray]:
    B2P_R1 = np.array([[1,0,0],[0,0,1],[0,-1,0]], dtype=np.float64)
    B2P_R2 = np.array([[-1,0,0],[0,1,0],[0,0,-1]], dtype=np.float64)
    B2P_T  = np.array([[-1,0,0],[0,0,-1],[0,1,0]], dtype=np.float64)
    R = B2P_R1 @ B[:3, :3] @ B2P_R2
    T = B2P_T @ B[:3, 3] @ R

    return R, T


