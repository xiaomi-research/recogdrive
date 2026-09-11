from typing import List, Optional, Tuple,Dict, Any

import cv2
import numpy as np
import numpy.typing as npt
from PIL import ImageColor
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
import matplotlib.patches as patches
import torch
from navsim.common.dataclasses import Camera, Lidar, Annotations, Trajectory
from navsim.common.enums import LidarIndex, BoundingBoxIndex
from navsim.visualization.config import AGENT_CONFIG
from navsim.visualization.lidar import filter_lidar_pc, get_lidar_pc_color
from navsim.planning.scenario_builder.navsim_scenario_utils import tracked_object_types
import os
import matplotlib

def add_camera_ax(ax: plt.Axes, camera: Camera) -> plt.Axes:
    """
    Adds camera image to matplotlib ax object
    :param ax: matplotlib ax object
    :param camera: navsim camera dataclass
    :return: ax object with image
    """
    ax.imshow(camera.image)
    return ax


def add_lidar_to_camera_ax(ax: plt.Axes, camera: Camera, lidar: Lidar) -> plt.Axes:
    """
    Adds camera image with lidar point cloud on matplotlib ax object
    :param ax: matplotlib ax object
    :param camera: navsim camera dataclass
    :param lidar: navsim lidar dataclass
    :return: ax object with image
    """

    image, lidar_pc = camera.image.copy(), lidar.lidar_pc.copy()
    image_height, image_width = image.shape[:2]
    print(lidar_pc.shape)
    lidar_pc = filter_lidar_pc(lidar_pc)
    lidar_pc_colors = np.array(get_lidar_pc_color(lidar_pc))

    pc_in_cam, pc_in_fov_mask = _transform_pcs_to_images(
        lidar_pc,
        camera.sensor2lidar_rotation,
        camera.sensor2lidar_translation,
        camera.intrinsics,
        img_shape=(image_height, image_width),
    )
    print(image_height, image_width)
    pc_in_cam = pc_in_cam[pc_in_fov_mask]
    print(pc_in_cam.shape)
    for (x, y), color in zip(pc_in_cam, lidar_pc_colors[pc_in_fov_mask]):
        color = (int(color[0]), int(color[1]), int(color[2]))
        cv2.circle(image, (int(x), int(y)), 5, color, -1)

    ax.imshow(image)
    return ax

def dense_map(Pts, n, m, grid):
    """
    Generate a dense depth map from the sparse points.
    :param Pts: The sparse depth points (x, y, depth)
    :param n: Image width
    :param m: Image height
    :param grid: Neighborhood grid size for interpolation
    :return: Dense depth map
    """
    ng = 2 * grid + 1
    mX = np.zeros((m, n)) + np.float("inf")
    mY = np.zeros((m, n)) + np.float("inf")
    mD = np.zeros((m, n))
    
    # Fill the sparse depth points into the mX, mY, and mD matrices
    mX[np.int32(Pts[1]), np.int32(Pts[0])] = Pts[0] - np.round(Pts[0])
    mY[np.int32(Pts[1]), np.int32(Pts[0])] = Pts[1] - np.round(Pts[1])
    mD[np.int32(Pts[1]), np.int32(Pts[0])] = Pts[2]
    
    KmX = np.zeros((ng, ng, m - ng, n - ng))
    KmY = np.zeros((ng, ng, m - ng, n - ng))
    KmD = np.zeros((ng, ng, m - ng, n - ng))
    
    for i in range(ng):
        for j in range(ng):
            KmX[i, j] = mX[i: (m - ng + i), j: (n - ng + j)] - grid - 1 + i
            KmY[i, j] = mY[i: (m - ng + i), j: (n - ng + j)] - grid - 1 + i
            KmD[i, j] = mD[i: (m - ng + i), j: (n - ng + j)]
    
    S = np.zeros_like(KmD[0, 0])
    Y = np.zeros_like(KmD[0, 0])
    
    for i in range(ng):
        for j in range(ng):
            s = 1 / np.sqrt(KmX[i, j] * KmX[i, j] + KmY[i, j] * KmY[i, j])
            Y = Y + s * KmD[i, j]
            S = S + s
    
    S[S == 0] = 1
    out = np.zeros((m, n))
    out[grid + 1: -grid, grid + 1: -grid] = Y / S
    return out

def add_lidar_to_camera_ax_with_depth(
    ax: plt.Axes, camera: Camera, lidar: Lidar
) -> Tuple[plt.Axes, npt.NDArray[np.float32]]:
    """
    Adds camera image with lidar point cloud on matplotlib ax object and generates depth map.
    :param ax: matplotlib ax object
    :param camera: navsim camera dataclass
    :param lidar: navsim lidar dataclass
    :return: ax object with image and the dense depth map
    """
    
    # 获取图像和LIDAR点云
    image, lidar_pc = camera.image.copy(), lidar.lidar_pc.copy()
    image_height, image_width = image.shape[:2]
    print("LIDAR point cloud shape:", lidar_pc.shape)

    lidar_pc = filter_lidar_pc(lidar_pc)
    lidar_pc_colors = np.array(get_lidar_pc_color(lidar_pc))

    pc_in_cam, pc_in_fov_mask, depth_values = _transform_pcs_to_images_with_depth(
        lidar_pc,
        camera.sensor2lidar_rotation,
        camera.sensor2lidar_translation,
        camera.intrinsics,
        img_shape=(image_height, image_width),
    )

    print("Image dimensions:", image_height, image_width)
    print("Number of points in field of view:", np.sum(pc_in_fov_mask))

    # 只保留视野内的点
    pc_in_cam = pc_in_cam[pc_in_fov_mask]
    depth_values = depth_values[pc_in_fov_mask]  # 只保留视野内的深度值
    lidar_pc_colors = lidar_pc_colors[pc_in_fov_mask]  # 只保留视野内的颜色

    # 创建空的深度图，初始化为无穷远值（大于可能的最大深度）
    depth_map_intermediate = np.full((image_height, image_width), np.inf)

    # 将 LIDAR 点投影到图像平面并绘制
    for (x, y), depth, color in zip(pc_in_cam[:, 0:2], depth_values, lidar_pc_colors):  # 只提取 (x, y)
        color = (int(color[0]), int(color[1]), int(color[2]))  # 转换为整数颜色
        x, y = int(x), int(y)
        if 0 <= x < image_width and 0 <= y < image_height:
            color = (int(color[0]), int(color[1]), int(color[2]))
            cv2.circle(image, (int(x), int(y)), 5, color, -1)
            depth_map_intermediate[y, x] = min(depth_map_intermediate[y, x], depth)  # 更新深度图

    ax.imshow(image)

    # 打印深度图的最小值和最大值，检查其范围
    valid_depth_values = depth_map_intermediate[depth_map_intermediate != np.inf]

    print("Depth map min value:", np.min(valid_depth_values))
    print("Depth map max value:", np.max(valid_depth_values))

    # 处理 np.inf 值：将 np.inf 替换为深度图中的最大有效深度
    if len(valid_depth_values) > 0:
        max_depth = np.max(valid_depth_values)
        depth_map_intermediate[np.isinf(depth_map_intermediate)] = max_depth
    else:
        max_depth = 0  # 如果没有有效的深度值，则使用 0 作为最大深度值
        depth_map_intermediate[np.isinf(depth_map_intermediate)] = max_depth

    # 使用 dense_map 函数生成密集深度图
    dense_depth_map = dense_map(
        np.array([pc_in_cam[:, 0], pc_in_cam[:, 1], depth_values]),  # x, y, depth
        image_width,
        image_height,
        grid=8  # 你可以调整这个值来控制平滑度
    )

    # 归一化深度图到 0-255 范围
    dense_depth_map_normalized = cv2.normalize(dense_depth_map, None, 0, 255, cv2.NORM_MINMAX)

    # 使用 'Spectral_r' 颜色映射应用于密集深度图
    colormap = plt.get_cmap('Spectral_r')
    colored_dense_depth_map = colormap(dense_depth_map_normalized / 255.0)  # 归一化后映射到 [0, 1]

    # 将 rgba 转换为 rgb
    colored_dense_depth_map = (colored_dense_depth_map[:, :, :3] * 255).astype(np.uint8)

    # 获取文件名和输出目录
    outdir = './output_dense_depth_maps'
    os.makedirs(outdir, exist_ok=True)
    filename = "lidar_dense_depth_image"  # 这里可以用你自己的文件名

    # 保存密集深度图
    cv2.imwrite(os.path.join(outdir, filename + '_dense_depth.png'), colored_dense_depth_map)

    # 合并图像和密集深度图
    split_region = np.ones((image.shape[0], 50, 3), dtype=np.uint8) * 255
    combined_result = cv2.hconcat([image, split_region, colored_dense_depth_map])

    # 保存合并结果
    cv2.imwrite(os.path.join(outdir, filename + '_combined.png'), combined_result)

    return ax, dense_depth_map

def add_annotations_to_camera_ax(ax: plt.Axes, camera: Camera, annotations: Annotations) -> plt.Axes:
    """
    Adds camera image with bounding boxes on matplotlib ax object
    :param ax: matplotlib ax object
    :param camera: navsim camera dataclass
    :param annotations: navsim annotations dataclass
    :return: ax object with image
    """

    box_labels = annotations.names
    boxes = _transform_annotations_to_camera(
        annotations.boxes,
        camera.sensor2lidar_rotation,
        camera.sensor2lidar_translation,
    )
    box_positions, box_dimensions, box_heading = (
        boxes[:, BoundingBoxIndex.POSITION],
        boxes[:, BoundingBoxIndex.DIMENSION],
        boxes[:, BoundingBoxIndex.HEADING],
    )
    corners_norm = np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)
    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array([0.5, 0.5, 0.5])
    corners = box_dimensions.reshape([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])
    
    corners = _rotation_3d_in_axis(corners, box_heading, axis=1)
    corners += box_positions.reshape(-1, 1, 3)

    # Then draw project corners to image.
    box_corners, corners_pc_in_fov = _transform_points_to_image(corners.reshape(-1, 3), camera.intrinsics)
    box_corners = box_corners.reshape(-1, 8, 2)
    corners_pc_in_fov = corners_pc_in_fov.reshape(-1, 8)
    valid_corners = corners_pc_in_fov.any(-1)

    box_corners, box_labels = box_corners[valid_corners], box_labels[valid_corners]
    image = _plot_rect_3d_on_img(camera.image.copy(), box_corners, box_labels)

    ax.imshow(image)
    return ax

def add_2d_annotations_to_camera_ax(ax: plt.Axes, camera: Camera, annotations: Annotations) -> plt.Axes:
    """
    Adds camera image with 2D bounding boxes on matplotlib ax object.
    :param ax: matplotlib ax object
    :param camera: navsim camera dataclass
    :param annotations: navsim annotations dataclass
    :return: ax object with image
    """

    box_labels = annotations.names
    boxes = _transform_annotations_to_camera(
        annotations.boxes,
        camera.sensor2lidar_rotation,
        camera.sensor2lidar_translation,
    )
    box_positions, box_dimensions, box_heading = (
        boxes[:, BoundingBoxIndex.POSITION],
        boxes[:, BoundingBoxIndex.DIMENSION],
        boxes[:, BoundingBoxIndex.HEADING],
    )
    corners_norm = np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)
    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array([0.5, 0.5, 0.5])
    corners = box_dimensions.reshape([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])

    corners = _rotation_3d_in_axis(corners, box_heading, axis=1)
    corners += box_positions.reshape(-1, 1, 3)

    # Project corners to image.
    box_corners, corners_pc_in_fov = _transform_points_to_image(corners.reshape(-1, 3), camera.intrinsics)
    box_corners = box_corners.reshape(-1, 8, 2)
    corners_pc_in_fov = corners_pc_in_fov.reshape(-1, 8)
    valid_corners = corners_pc_in_fov.any(-1)

    box_corners, box_labels = box_corners[valid_corners], box_labels[valid_corners]

    # Calculate 2D bounding boxes from projected 3D corners.
    box_2d_list = []
    for corner_set in box_corners:
        min_x = np.min(corner_set[:, 0])
        max_x = np.max(corner_set[:, 0])
        min_y = np.min(corner_set[:, 1])
        max_y = np.max(corner_set[:, 1])
        box_2d_list.append([min_x, min_y, max_x, max_y])

    box_2d_list = np.array(box_2d_list)
    print(box_2d_list)
    image = _plot_rect_2d_on_img(camera.image.copy(), box_2d_list, box_labels)

    ax.imshow(image)
    return ax

def _plot_rect_2d_on_img(image: np.ndarray, boxes_2d: np.ndarray, labels: list, color=(0, 255, 0), thickness=2) -> np.ndarray:
    """
    Draws 2D bounding boxes on an image.
    :param image: input image (numpy array)
    :param boxes_2d: 2D bounding boxes (numpy array), each box is [x_min, y_min, x_max, y_max]
    :param labels: list of labels for each bounding box
    :param color: color of the bounding box (BGR format)
    :param thickness: thickness of the bounding box lines
    :return: image with bounding boxes drawn
    """

    image_copy = image.copy()

    if boxes_2d is not None and len(boxes_2d) > 0:
        # if labels is None:
        #   labels = []
        for i, box in enumerate(boxes_2d):
            x_min, y_min, x_max, y_max = map(int, box)  # Convert coordinates to integers
            cv2.rectangle(image_copy, (x_min, y_min), (x_max, y_max), color, thickness)

            # # Add label
            # if labels and i < len(labels):
            #     label = labels[i]
            #     cv2.putText(image_copy, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

    return image_copy


def _transform_annotations_to_camera(
    boxes: npt.NDArray[np.float32],
    sensor2lidar_rotation: npt.NDArray[np.float32],
    sensor2lidar_translation: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """
    Helper function to transform bounding boxes into camera frame
    TODO: Refactor
    :param boxes: array representation of bounding boxes
    :param sensor2lidar_rotation: camera rotation
    :param sensor2lidar_translation: camera translation
    :return: bounding boxes in camera coordinates
    """

    locs, rots = (
        boxes[:, BoundingBoxIndex.POSITION],
        boxes[:, BoundingBoxIndex.HEADING :],
    )
    dims_cam = boxes[
        :, [BoundingBoxIndex.LENGTH, BoundingBoxIndex.HEIGHT, BoundingBoxIndex.WIDTH]
    ]  # l, w, h -> l, h, w

    rots_cam = np.zeros_like(rots)
    for idx, rot in enumerate(rots):
        rot = Quaternion(axis=[0, 0, 1], radians=rot)
        rot = Quaternion(matrix=sensor2lidar_rotation).inverse * rot
        rots_cam[idx] = -rot.yaw_pitch_roll[0]

    lidar2cam_r = np.linalg.inv(sensor2lidar_rotation)
    lidar2cam_t = sensor2lidar_translation @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t

    locs_cam = np.concatenate([locs, np.ones_like(locs)[:, :1]], -1)  # -1, 4
    locs_cam = lidar2cam_rt.T @ locs_cam.T
    locs_cam = locs_cam.T
    locs_cam = locs_cam[:, :-1]
    return np.concatenate([locs_cam, dims_cam, rots_cam], -1)


def _rotation_3d_in_axis(points: npt.NDArray[np.float32], angles: npt.NDArray[np.float32], axis: int = 0):
    """
    Rotate 3D points by angles according to axis.
    TODO: Refactor
    :param points: array of points
    :param angles: array of angles
    :param axis: axis to perform rotation, defaults to 0
    :raises value: _description_
    :raises ValueError: if axis invalid
    :return: rotated points
    """
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    ones = np.ones_like(rot_cos)
    zeros = np.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = np.stack(
            [
                np.stack([rot_cos, zeros, -rot_sin]),
                np.stack([zeros, ones, zeros]),
                np.stack([rot_sin, zeros, rot_cos]),
            ]
        )
    elif axis == 2 or axis == -1:
        rot_mat_T = np.stack(
            [
                np.stack([rot_cos, -rot_sin, zeros]),
                np.stack([rot_sin, rot_cos, zeros]),
                np.stack([zeros, zeros, ones]),
            ]
        )
    elif axis == 0:
        rot_mat_T = np.stack(
            [
                np.stack([zeros, rot_cos, -rot_sin]),
                np.stack([zeros, rot_sin, rot_cos]),
                np.stack([ones, zeros, zeros]),
            ]
        )
    else:
        raise ValueError(f"axis should in range [0, 1, 2], got {axis}")
    return np.einsum("aij,jka->aik", points, rot_mat_T)


def _plot_rect_3d_on_img(
    image: npt.NDArray[np.float32],
    box_corners: npt.NDArray[np.float32],
    box_labels: List[str],
    thickness: int = 3,
) -> npt.NDArray[np.uint8]:
    """
    Plot the boundary lines of 3D rectangular on 2D images.
    TODO: refactor
    :param image:  The numpy array of image.
    :param box_corners: Coordinates of the corners of 3D, shape of [N, 8, 2].
    :param box_labels: labels of boxes for coloring
    :param thickness: pixel width of liens, defaults to 3
    :return: image with 3D bounding boxes
    """
    line_indices = (
        (0, 1),
        (0, 3),
        (0, 4),
        (1, 2),
        (1, 5),
        (3, 2),
        (3, 7),
        (4, 5),
        (4, 7),
        (2, 6),
        (5, 6),
        (6, 7),
    )
    for i in range(len(box_corners)):
        layer = tracked_object_types[box_labels[i]]
        color = ImageColor.getcolor(AGENT_CONFIG[layer]["fill_color"], "RGB")
        corners = box_corners[i].astype(int)
        for start, end in line_indices:
            cv2.line(
                image,
                (corners[start, 0], corners[start, 1]),
                (corners[end, 0], corners[end, 1]),
                color,
                thickness,
                cv2.LINE_AA,
            )
    return image.astype(np.uint8)


def _transform_points_to_image(
    points: npt.NDArray[np.float32],
    intrinsic: npt.NDArray[np.float32],
    image_shape: Optional[Tuple[int, int]] = None,
    eps: float = 1e-3,
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """
    Transforms points in camera frame to image pixel coordinates
    TODO: refactor
    :param points: points in camera frame
    :param intrinsic: camera intrinsics
    :param image_shape: shape of image in pixel
    :param eps: lower threshold of points, defaults to 1e-3
    :return: points in pixel coordinates, mask of values in frame
    """
    points = points[:, :3]

    viewpad = np.eye(4)
    viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic

    pc_img = np.concatenate([points, np.ones_like(points)[:, :1]], -1)
    pc_img = viewpad @ pc_img.T
    pc_img = pc_img.T

    cur_pc_in_fov = pc_img[:, 2] > eps
    pc_img = pc_img[..., 0:2] / np.maximum(pc_img[..., 2:3], np.ones_like(pc_img[..., 2:3]) * eps)
    if image_shape is not None:
        img_h, img_w = image_shape
        cur_pc_in_fov = (
            cur_pc_in_fov
            & (pc_img[:, 0] < (img_w - 1))
            & (pc_img[:, 0] > 0)
            & (pc_img[:, 1] < (img_h - 1))
            & (pc_img[:, 1] > 0)
        )
    return pc_img, cur_pc_in_fov


def _transform_pcs_to_images(
    lidar_pc: npt.NDArray[np.float32],
    sensor2lidar_rotation: npt.NDArray[np.float32],
    sensor2lidar_translation: npt.NDArray[np.float32],
    intrinsic: npt.NDArray[np.float32],
    img_shape: Optional[Tuple[int, int]] = None,
    eps: float = 1e-3,
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """
    Transforms points in camera frame to image pixel coordinates
    TODO: refactor
    :param lidar_pc: lidar point cloud
    :param sensor2lidar_rotation: camera rotation
    :param sensor2lidar_translation: camera translation
    :param intrinsic: camera intrinsics
    :param img_shape: image shape in pixels, defaults to None
    :param eps: threshold for lidar pc height, defaults to 1e-3
    :return: lidar pc in pixel coordinates, mask of values in frame
    """
    pc_xyz = lidar_pc[LidarIndex.POSITION, :].T

    lidar2cam_r = np.linalg.inv(sensor2lidar_rotation)
    lidar2cam_t = sensor2lidar_translation @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t

    viewpad = np.eye(4)
    viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
    lidar2img_rt = viewpad @ lidar2cam_rt.T

    cur_pc_xyz = np.concatenate([pc_xyz, np.ones_like(pc_xyz)[:, :1]], -1)
    cur_pc_cam = lidar2img_rt @ cur_pc_xyz.T
    cur_pc_cam = cur_pc_cam.T
    cur_pc_in_fov = cur_pc_cam[:, 2] > eps
    cur_pc_cam = cur_pc_cam[..., 0:2] / np.maximum(cur_pc_cam[..., 2:3], np.ones_like(cur_pc_cam[..., 2:3]) * eps)

    if img_shape is not None:
        img_h, img_w = img_shape
        cur_pc_in_fov = (
            cur_pc_in_fov
            & (cur_pc_cam[:, 0] < (img_w - 1))
            & (cur_pc_cam[:, 0] > 0)
            & (cur_pc_cam[:, 1] < (img_h - 1))
            & (cur_pc_cam[:, 1] > 0)
        )
    return cur_pc_cam, cur_pc_in_fov

def _transform_pcs_to_images_with_depth(
    lidar_pc: npt.NDArray[np.float32],
    sensor2lidar_rotation: npt.NDArray[np.float32],
    sensor2lidar_translation: npt.NDArray[np.float32],
    intrinsic: npt.NDArray[np.float32],
    img_shape: Optional[Tuple[int, int]] = None,
    eps: float = 1e-3,
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_], npt.NDArray[np.float32]]:
    """
    Transforms points in camera frame to image pixel coordinates and returns depth values.
    :param lidar_pc: lidar point cloud
    :param sensor2lidar_rotation: camera rotation
    :param sensor2lidar_translation: camera translation
    :param intrinsic: camera intrinsics
    :param img_shape: image shape in pixels, defaults to None
    :param eps: threshold for lidar pc height, defaults to 1e-3
    :return: lidar pc in pixel coordinates, mask of values in frame, depth values in frame
    """
    # 获取 LIDAR 点云的 (x, y, z) 坐标
    pc_xyz = lidar_pc[LidarIndex.POSITION, :].T  # shape should be (n, 3), each row [x, y, z]
    print(f"pc_xyz shape: {pc_xyz.shape}")  # Debug: Check shape of lidar point cloud

    lidar2cam_r = np.linalg.inv(sensor2lidar_rotation)
    lidar2cam_t = sensor2lidar_translation @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t

    viewpad = np.eye(4)
    viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
    lidar2img_rt = viewpad @ lidar2cam_rt.T

    # 添加齐次坐标，确保是 4D 向量 (x, y, z, 1)
    cur_pc_xyz = np.concatenate([pc_xyz, np.ones_like(pc_xyz)[:, :1]], axis=-1)  # shape: (n, 4)
    print(f"cur_pc_xyz shape after concat: {cur_pc_xyz.shape}")  # Debug: Check shape after concat

    # 投影到相机坐标系
    cur_pc_cam = lidar2img_rt @ cur_pc_xyz.T  # 投影矩阵 * 点云
    cur_pc_cam = cur_pc_cam.T  # 转置回来 (n, 4)
    print(f"cur_pc_cam shape after projection: {cur_pc_cam.shape}")  # Debug: Check shape after projection

    # 过滤掉视野外的点
    cur_pc_in_fov = cur_pc_cam[:, 2] > eps  # z > eps 才是有效点

    # 进行齐次坐标的标准化：通过 Z 值标准化 (x, y)，得到二维坐标
    cur_pc_cam[:, 0:2] = cur_pc_cam[:, 0:2] / np.maximum(cur_pc_cam[:, 2:3], np.ones_like(cur_pc_cam[:, 2:3]) * eps)
    print(f"cur_pc_cam shape after normalization: {cur_pc_cam.shape}")  # Debug: Check shape after normalization

    # 获取深度值（即 Z 坐标）
    depth_values = cur_pc_cam[:, 2]  # 深度值为 Z 坐标
    print(f"depth_values shape: {depth_values.shape}")  # Debug: Check depth_values shape

    # 只保留在视野内的点
    if img_shape is not None:
        img_h, img_w = img_shape
        cur_pc_in_fov = (
            cur_pc_in_fov
            & (cur_pc_cam[:, 0] < (img_w - 1))
            & (cur_pc_cam[:, 0] > 0)
            & (cur_pc_cam[:, 1] < (img_h - 1))
            & (cur_pc_cam[:, 1] > 0)
        )

    return cur_pc_cam, cur_pc_in_fov, depth_values



def add_trajectory_to_camera_ax(ax: plt.Axes, camera: Camera, trajectory: Trajectory, config: Dict[str, Any]) -> plt.Axes:
    """
    将轨迹姿势作为连接的线条添加到相机图像上，并在轨迹最前方绘制箭头。
    如果轨迹的第一个点起始于图像外，则计算轨迹线(由第一和第二个点定义)与图像边界的交点作为起始点。
    :param ax: matplotlib ax 对象
    :param camera: navsim camera dataclass
    :param trajectory: navsim trajectory dataclass
    :param config: 包含绘图参数的字典
    :return: ax with plot
    """
    poses_2d = trajectory.poses[:, :2]  # 获取 x, y，假设 trajectory.poses 是 [x, y, heading]
    poses_3d = np.concatenate([poses_2d, np.zeros((poses_2d.shape[0], 1))], axis=1)  # 添加 z=0 列
    poses = np.concatenate([np.array([[0, 0, 0]]), poses_3d])  # 假设轨迹起始点在 (0,0,0)

    pc_in_cam, pc_in_fov_mask = _transform_pcs_to_images(
        poses.T,
        camera.sensor2lidar_rotation,
        camera.sensor2lidar_translation,
        camera.intrinsics,
        img_shape=camera.image.shape[:2]
    )

    image_height, image_width = camera.image.shape[:2]
    valid_cam_points = pc_in_cam[pc_in_fov_mask]

    points_to_plot = []
    first_point = pc_in_cam[1] if len(pc_in_cam) > 1 else None # 第一个轨迹点投影后的点
    second_point = pc_in_cam[2] if len(pc_in_cam) > 2 else None # 第二个轨迹点投影后的点

    # 检查第一个轨迹点是否在图像内
    if first_point is not None and pc_in_fov_mask[1]:
        # 如果第一个轨迹点在图像内，则直接使用第一个轨迹点作为起始点
        points_to_plot.append(first_point)
    elif first_point is not None and second_point is not None:
        # 如果第一个轨迹点在图像外，且有第二个轨迹点，计算交点
        start_pt = first_point
        end_pt = second_point
        intersection_point = _get_intersection_with_image_bottom_boundary(start_pt, end_pt, image_width, image_height)
        if intersection_point is not None:
            points_to_plot.append(intersection_point) # 使用交点作为起始点

    # 添加剩余的轨迹点 (从第二个点之后开始，因为第一个点或交点已经处理)
    for i in range(2, len(pc_in_cam)): # 从索引 2 开始
        if pc_in_fov_mask[i]:
            points_to_plot.append(pc_in_cam[i])

    if len(points_to_plot) > 1:  # 确保至少有两个点来画线
        plot_points = np.array(points_to_plot)
        # 将有效的相机坐标点连接成线
        ax.plot(
            plot_points[:, 0],
            plot_points[:, 1],
            color=config["line_color"],
            alpha=config["line_color_alpha"],
            linewidth=config["line_width"],
            linestyle=config["line_style"],
            marker=config.get("marker", None),  # 允许配置是否显示marker，默认为不显示
            markersize=config.get("marker_size", 0),
            markeredgecolor=config.get("marker_edge_color", None),
            zorder=config["zorder"],  # 确保轨迹线在图像之上
        )

        # # 添加箭头，指向轨迹的最后一个线段，以最后一点为起点 (使用 points_to_plot)
        last_point = plot_points[-1]
        second_last_point = plot_points[-2]

        arrow_dx = last_point[0] - second_last_point[0]
        arrow_dy = last_point[1] - second_last_point[1]

        # 计算箭头终点，在最后一个点的基础上，沿轨迹方向延伸一段距离 (例如，方向向量的长度)
        arrow_end_x = last_point[0] + arrow_dx
        arrow_end_y = last_point[1] + arrow_dy

        arrow = patches.FancyArrowPatch(
            posA=(last_point[0], last_point[1]),  # 箭头起点设置为最后一个轨迹点
            posB=(arrow_end_x, arrow_end_y),  # 箭头终点为在最后一点基础上延伸的点
            arrowstyle='-|>',  # 使用实心箭头
            mutation_scale=15,  # 箭头大小，可以调整
            fc=config["arrow_color"],  # 箭头颜色
            ec=config["arrow_edge_color"],  # 箭头边缘颜色
            alpha=config["arrow_alpha"],  # 箭头透明度
            linewidth=config.get("arrow_line_width", config["line_width"]),  # 箭头线的宽度，默认与轨迹线宽相同
            connectionstyle='arc3,rad=0.0',
            zorder=config["zorder"] + 1,  # 箭头zorder比线高一点，确保在最前面
        )
        ax.add_patch(arrow)

    return ax


# from scipy.interpolate import splprep, splev

# def add_trajectory_to_camera_ax(ax: plt.Axes, camera: Camera, trajectory: Trajectory, config: Dict[str, Any]) -> plt.Axes:
#     poses_2d = trajectory.poses[:, :2]
#     poses_3d = np.concatenate([poses_2d, np.zeros((poses_2d.shape[0], 1))], axis=1)
#     poses = np.concatenate([np.array([[0, 0, 0]]), poses_3d])

#     pc_in_cam, pc_in_fov_mask = _transform_pcs_to_images(
#         poses.T,
#         camera.sensor2lidar_rotation,
#         camera.sensor2lidar_translation,
#         camera.intrinsics,
#         img_shape=camera.image.shape[:2]
#     )

#     image_height, image_width = camera.image.shape[:2]

#     # === 起点处理：保证轨迹从视野内或边界进入 ===
#     points_to_plot = []
#     first_point = pc_in_cam[1] if len(pc_in_cam) > 1 else None
#     second_point = pc_in_cam[2] if len(pc_in_cam) > 2 else None

#     if first_point is not None and pc_in_fov_mask[1]:
#         points_to_plot.append(first_point)
#     elif first_point is not None and second_point is not None:
#         intersection_point = _get_intersection_with_image_bottom_boundary(
#             first_point, second_point, image_width, image_height
#         )
#         if intersection_point is not None:
#             points_to_plot.append(intersection_point)

#     for i in range(2, len(pc_in_cam)):
#         if pc_in_fov_mask[i]:
#             points_to_plot.append(pc_in_cam[i])

#     valid_points = np.array(points_to_plot)

#     if len(valid_points) < 2:
#         return ax

#     # === spline 平滑轨迹 ===
#     x, y = valid_points[:, 0], valid_points[:, 1]
#     try:
#         tck, u = splprep([x, y], s=2)  # s 控制平滑程度
#         u_fine = np.linspace(0, 1, 200)
#         x_smooth, y_smooth = splev(u_fine, tck)
#     except Exception:
#         # 如果点数不足以 spline，就直接连线
#         x_smooth, y_smooth = x, y

#     # === 干净的线条（无 glow，无箭头） ===
#     ax.plot(x_smooth, y_smooth,
#             color=config["line_color"],
#             alpha=config["line_color_alpha"],
#             linewidth=config["line_width"],
#             linestyle=config["line_style"],
#             zorder=config["zorder"])

#     return ax



def _get_intersection_with_image_bottom_boundary(origin_point, end_point, image_width, image_height):
    """
    计算线段与图像*底部*边界的交点。
    :param origin_point: 线段的起始点 (numpy array [x, y])
    :param end_point: 线段的结束点 (numpy array [x, y])
    :param image_width: 图像宽度
    :param image_height: 图像高度
    :return: 线段与图像底部边界的交点 (numpy array [x, y])，如果没有交点或起始点在图像内，返回 None
    """
    x1, y1 = origin_point
    x2, y2 = end_point

    # 图像底部边界 y = y_max
    x_min, x_max = 0, image_width - 1
    y_max = image_height - 1


    # 1. 与下边界 y = y_max 的交点
    if y1 != y2: # 避免除以零
        x_intersection_bottom = x1 + (y_max - y1) * (x2 - x1) / (y2 - y1)
        if x_min <= x_intersection_bottom <= x_max and min(y1, y2) <= y_max <= max(y1, y2): # 检查交点x坐标在图像宽度内，且y_max在y1, y2之间
            return np.array([x_intersection_bottom, y_max])

    return None # 没有与底部边界的有效交点
