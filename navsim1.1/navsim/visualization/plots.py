from typing import Any, Callable, List, Tuple
import io

from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from navsim.evaluate.pdm_score import pdm_score

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import Scene
from navsim.visualization.config import BEV_PLOT_CONFIG, TRAJECTORY_CONFIG, CAMERAS_PLOT_CONFIG
from navsim.visualization.bev import add_configured_bev_on_ax, add_trajectory_to_bev_ax
from navsim.visualization.camera import add_annotations_to_camera_ax, add_lidar_to_camera_ax, add_camera_ax,add_trajectory_to_camera_ax


def configure_bev_ax(ax: plt.Axes) -> plt.Axes:
    """
    Configure the plt ax object for birds-eye-view plots
    :param ax: matplotlib ax object
    :return: configured ax object
    """

    margin_x, margin_y = BEV_PLOT_CONFIG["figure_margin"]
    ax.set_aspect("equal")

    # NOTE: x forward, y sideways
    ax.set_xlim(-margin_y / 2, margin_y / 2)
    ax.set_ylim(-margin_x / 2, margin_x / 2)

    # NOTE: left is y positive, right is y negative
    ax.invert_xaxis()

    return ax


def configure_ax(ax: plt.Axes) -> plt.Axes:
    """
    Configure the ax object for general plotting
    :param ax: matplotlib ax object
    :return: ax object without a,y ticks
    """
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def configure_all_ax(ax: List[List[plt.Axes]]) -> List[List[plt.Axes]]:
    """
    Iterates through 2D ax list/array to apply configurations
    :param ax: 2D list/array of matplotlib ax object
    :return: configure axes
    """
    for i in range(len(ax)):
        for j in range(len(ax[i])):
            configure_ax(ax[i][j])

    return ax


def plot_bev_frame(scene: Scene, frame_idx: int) -> Tuple[plt.Figure, plt.Axes]:
    """
    General plot for birds-eye-view visualization
    :param scene: navsim scene dataclass
    :param frame_idx: index of selected frame
    :return: figure and ax object of matplotlib
    """
    fig, ax = plt.subplots(1, 1, figsize=BEV_PLOT_CONFIG["figure_size"])
    add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])
    configure_bev_ax(ax)
    configure_ax(ax)

    return fig, ax


def plot_bev_with_agent(scene: Scene, scene_traj: Scene,agent: AbstractAgent) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots agent and human trajectory in birds-eye-view visualization
    :param scene: navsim scene dataclass
    :param agent: navsim agent
    :return: figure and ax object of matplotlib
    """

    human_trajectory = scene.get_future_trajectory()
    agent_trajectory = agent.compute_trajectory_vis(scene_traj.get_agent_input())

    frame_idx = scene.scene_metadata.num_history_frames - 1
    fig, ax = plt.subplots(1, 1, figsize=BEV_PLOT_CONFIG["figure_size"])
    add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])
    add_trajectory_to_bev_ax(ax, human_trajectory, TRAJECTORY_CONFIG["human"])
    add_trajectory_to_bev_ax(ax, agent_trajectory, TRAJECTORY_CONFIG["agent"])
    configure_bev_ax(ax)
    configure_ax(ax)

    return fig, ax


def plot_cameras_frame_with_agent(
    scene: Scene, 
    scene_traj: Scene, 
    frame_idx: int, 
    agent: AbstractAgent
) -> Tuple[plt.Figure, List[List[plt.Axes]], plt.Axes]:
    """
    Plots 8x cameras in 3x3 grid (left) and birds-eye-view with trajectories (right)
    
    :param scene: navsim scene dataclass
    :param frame_idx: index of selected frame
    :param agent: navsim agent
    :return: (figure, 3x3 camera axes, bev axis)
    """
    frame = scene.frames[frame_idx]
    
    # 创建figure和网格布局
    fig = plt.figure(figsize=(24, 12))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.02)

    # 左侧3x3摄像头布局
    left_gs = gs[0].subgridspec(3, 3)
    left_ax = [[fig.add_subplot(left_gs[i, j]) for j in range(3)] for i in range(3)]
    
    # 填充摄像头（调整后的布局）
    add_camera_ax(left_ax[0][0], frame.cameras.cam_l0)
    add_camera_ax(left_ax[0][1], frame.cameras.cam_f0)
    
    add_camera_ax(left_ax[1][0], frame.cameras.cam_l1)
    add_camera_ax(left_ax[2][1], frame.cameras.cam_b0)  # 将后视摄像头移到中间
    add_camera_ax(left_ax[1][2], frame.cameras.cam_r1)
    
    add_camera_ax(left_ax[2][0], frame.cameras.cam_l2)
    left_ax[1][1].axis("off")  # 留空位置
    add_camera_ax(left_ax[2][2], frame.cameras.cam_r2)

    # 配置摄像头坐标轴
    for row in left_ax:
        for ax in row:
            ax.axis("off")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("auto")

    # 右侧BEV布局
    right_ax = fig.add_subplot(gs[1])
    
    # 添加BEV和轨迹
    add_configured_bev_on_ax(right_ax, scene.map_api, frame)
    human_trajectory = scene.get_future_trajectory()
    agent_trajectory = agent.compute_trajectory(scene_traj.get_agent_input())

    add_trajectory_to_bev_ax(ax, human_trajectory, TRAJECTORY_CONFIG["human"])
    add_trajectory_to_bev_ax(ax, agent_trajectory, TRAJECTORY_CONFIG["agent"])

    configure_bev_ax(right_ax)

    # 调整布局参数
    fig.tight_layout()
    fig.subplots_adjust(
        wspace=0.02, hspace=0.02,
        left=0.01, right=0.99,
        top=0.95, bottom=0.05
    )

    return fig, left_ax, right_ax


def plot_bev_and_camera_with_agent(
    scene: Scene,
    scene_traj: Scene,
    frame_idx: int,
    agent: AbstractAgent
) -> Tuple[plt.Figure, plt.Axes, plt.Axes]:
    """
    融合函数: 左侧可视化鸟瞰图 (BEV) 和轨迹，右侧可视化前视摄像头 (cam_f0) 和人类轨迹。

    :param scene: navsim scene dataclass
    :param scene_traj: scene trajectory dataclass
    :param frame_idx: index of selected frame
    :param agent: navsim agent
    :return: (figure, bev axis, camera axis)
    """
    frame = scene.frames[frame_idx]

    # 创建 figure 和 gridspec 布局，左右两列
    fig = plt.figure(figsize=(18, 9)) # 调整figure大小以适应左右布局
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.05) # 左右比例 1:1, 调整列间距

    # 左侧 BEV 布局
    bev_ax = fig.add_subplot(gs[1])

    # 添加 BEV 和轨迹 (人类和智能体)
    add_configured_bev_on_ax(bev_ax, scene.map_api, frame)
    human_trajectory = scene.get_future_trajectory()
    agent_trajectory = agent.compute_trajectory_vis(scene_traj.get_agent_input())

    add_trajectory_to_bev_ax(bev_ax, human_trajectory, TRAJECTORY_CONFIG["human"])
    add_trajectory_to_bev_ax(bev_ax, agent_trajectory, TRAJECTORY_CONFIG["agent"])

    configure_bev_ax(bev_ax) # 配置BEV坐标轴
    bev_ax.set_aspect("equal") # 确保BEV纵横比为1:1
    bev_ax.set_xticks([]) # 移除刻度
    bev_ax.set_yticks([])

    # 右侧前视摄像头布局
    camera_ax = fig.add_subplot(gs[0])

    # 只添加 f0 摄像头
    add_camera_ax(camera_ax, frame.cameras.cam_f0)

    # 在 f0 图上可视化人类轨迹
    #human_trajectory = scene.get_future_trajectory() # 再次获取人类轨迹，确保使用最新的
    trajectory_config = {  # 你可以根据需要调整这些参数
        "line_color": "red", # 设置线条颜色为红色
        "line_color_alpha": 0.7,
        "line_width": 4,
        "line_style": "-", # 设置线条样式为实线
        "marker": None,     # 不显示 marker
        "zorder": 3,
        "arrow_color": "red", # 设置箭头颜色为红色，与线条颜色一致
        "arrow_edge_color": "red",
        "arrow_alpha": 1.0,
        "arrow_line_width": 1.5,
    }
    camera_trajectory_config = trajectory_config # 复制人类轨迹配置

    add_trajectory_to_camera_ax(camera_ax, frame.cameras.cam_f0, agent_trajectory, camera_trajectory_config)

    human_trajectory = scene.get_future_trajectory()

    trajectory_config_human = {  # 你可以根据需要调整这些参数
        "line_color": "green", # 设置线条颜色为红色
        "line_color_alpha": 0.7,
        "line_width": 4,
        "line_style": "-", # 设置线条样式为实线
        "marker": None,     # 不显示 marker
        "zorder": 3,
        "arrow_color": "green", # 设置箭头颜色为红色，与线条颜色一致
        "arrow_edge_color": "green",
        "arrow_alpha": 1.0,
        "arrow_line_width": 1.5,
    }
    add_trajectory_to_camera_ax(camera_ax, frame.cameras.cam_f0, human_trajectory, trajectory_config_human)
    # 配置 f0 摄像头坐标轴
    camera_ax.axis("off") # 关闭坐标轴
    camera_ax.set_xticks([]) # 移除刻度
    camera_ax.set_yticks([])
    camera_ax.set_aspect("auto") # 根据图像内容自动调整纵横比


    # 调整整体布局参数
    fig.tight_layout() # 自动调整子图参数，使之填充整个figure区域
    fig.subplots_adjust(
        wspace=0.05, hspace=0.05, # 调整子图之间的水平和垂直间距
        left=0.03, right=0.97, # 调整左右边界
        top=0.95, bottom=0.05 # 调整上下边界
    )

    return fig, bev_ax, camera_ax

def plot_traj_with_agent(
    scene: Scene,
    scene_traj: Scene,
    frame_idx: int,
    agent: AbstractAgent,
    metric_cache,
    simulator,
    scorer
) -> Tuple[plt.Figure, List[List[plt.Axes]], plt.Axes]:
    frame = scene.frames[frame_idx]

    # 创建 figure 和 axes，只用于 f0 相机
    fig = plt.figure(figsize=(8, 6))  # 调整 figure 大小以适应单个相机
    ax_f0 = fig.add_subplot(1, 1, 1)  # 创建一个 subplot

    # 只添加 f0 摄像头
    add_camera_ax(ax_f0, frame.cameras.cam_f0)

    # 配置 f0 摄像头坐标轴
    ax_f0.axis("off")
    ax_f0.set_xticks([])
    ax_f0.set_yticks([])
    ax_f0.set_aspect("auto")

    # 获取 agent 轨迹
    agent_trajectory= agent.compute_trajectory_vis(scene_traj.get_agent_input())
    # 在 f0 图上可视化 agent_trajectory
    trajectory_config = {  # 你可以根据需要调整这些参数
        "line_color": "red", # 设置线条颜色为红色
        "line_color_alpha": 0.7,
        "line_width": 4,
        "line_style": "-", # 设置线条样式为实线
        "marker": None,     # 不显示 marker
        "zorder": 3,
        "arrow_color": "red", # 设置箭头颜色为红色，与线条颜色一致
        "arrow_edge_color": "red",
        "arrow_alpha": 1.0,
        "arrow_line_width": 1.5,
    }
    add_trajectory_to_camera_ax(ax_f0, frame.cameras.cam_f0, agent_trajectory, trajectory_config)

    pdm_result = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=agent_trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer
    )
    human_trajectory = scene.get_future_trajectory()

    trajectory_config_human = {  # 你可以根据需要调整这些参数
        "line_color": "green", # 设置线条颜色为红色
        "line_color_alpha": 0.7,
        "line_width": 4,
        "line_style": "-", # 设置线条样式为实线
        "marker": None,     # 不显示 marker
        "zorder": 3,
        "arrow_color": "green", # 设置箭头颜色为红色，与线条颜色一致
        "arrow_edge_color": "green",
        "arrow_alpha": 1.0,
        "arrow_line_width": 1.5,
    }
    add_trajectory_to_camera_ax(ax_f0, frame.cameras.cam_f0, human_trajectory, trajectory_config_human)

    fig.tight_layout()
    fig.subplots_adjust(
        left=0.05, right=0.95,
        top=0.95, bottom=0.05
    )

    return fig, [[ax_f0]], None,pdm_result,agent_trajectory # 返回包含 f0 ax 的列表和一个 None 值，代替 right_ax

def plot_cameras_frame(scene: Scene, frame_idx: int) -> Tuple[plt.Figure, Any]:
    """
    Plots 8x cameras and birds-eye-view visualization in 3x3 grid
    :param scene: navsim scene dataclass
    :param frame_idx: index of selected frame
    :return: figure and ax object of matplotlib
    """

    frame = scene.frames[frame_idx]
    fig, ax = plt.subplots(3, 3, figsize=CAMERAS_PLOT_CONFIG["figure_size"])

    add_camera_ax(ax[0, 0], frame.cameras.cam_l0)
    add_camera_ax(ax[0, 1], frame.cameras.cam_f0)
    add_camera_ax(ax[0, 2], frame.cameras.cam_r0)

    add_camera_ax(ax[1, 0], frame.cameras.cam_l1)
    add_configured_bev_on_ax(ax[1, 1], scene.map_api, frame)
    add_camera_ax(ax[1, 2], frame.cameras.cam_r1)

    add_camera_ax(ax[2, 0], frame.cameras.cam_l2)
    add_camera_ax(ax[2, 1], frame.cameras.cam_b0)
    add_camera_ax(ax[2, 2], frame.cameras.cam_r2)

    configure_all_ax(ax)
    configure_bev_ax(ax[1, 1])
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.01, hspace=0.01, left=0.01, right=0.99, top=0.99, bottom=0.01)

    return fig, ax


def plot_cameras_frame_with_lidar(scene: Scene, frame_idx: int) -> Tuple[plt.Figure, Any]:
    """
    Plots 8x cameras (including the lidar pc) and birds-eye-view visualization in 3x3 grid
    :param scene: navsim scene dataclass
    :param frame_idx: index of selected frame
    :return: figure and ax object of matplotlib
    """

    frame = scene.frames[frame_idx]
    fig, ax = plt.subplots(3, 3, figsize=CAMERAS_PLOT_CONFIG["figure_size"])

    add_lidar_to_camera_ax(ax[0, 0], frame.cameras.cam_l0, frame.lidar)
    add_lidar_to_camera_ax(ax[0, 1], frame.cameras.cam_f0, frame.lidar)
    add_lidar_to_camera_ax(ax[0, 2], frame.cameras.cam_r0, frame.lidar)

    add_lidar_to_camera_ax(ax[1, 0], frame.cameras.cam_l1, frame.lidar)
    add_configured_bev_on_ax(ax[1, 1], scene.map_api, frame)
    add_lidar_to_camera_ax(ax[1, 2], frame.cameras.cam_r1, frame.lidar)

    add_lidar_to_camera_ax(ax[2, 0], frame.cameras.cam_l2, frame.lidar)
    add_lidar_to_camera_ax(ax[2, 1], frame.cameras.cam_b0, frame.lidar)
    add_lidar_to_camera_ax(ax[2, 2], frame.cameras.cam_r2, frame.lidar)

    configure_all_ax(ax)
    configure_bev_ax(ax[1, 1])
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.01, hspace=0.01, left=0.01, right=0.99, top=0.99, bottom=0.01)

    return fig, ax


# def plot_cameras_frame_with_annotations(scene: Scene, frame_idx: int) -> Tuple[plt.Figure, Any]:
#     """
#     Plots 8x cameras (including the bounding boxes) and birds-eye-view visualization in 3x3 grid
#     :param scene: navsim scene dataclass
#     :param frame_idx: index of selected frame
#     :return: figure and ax object of matplotlib
#     """

#     frame = scene.frames[frame_idx]
#     fig, ax = plt.subplots(3, 3, figsize=CAMERAS_PLOT_CONFIG["figure_size"])

#     add_annotations_to_camera_ax(ax[0, 0], frame.cameras.cam_l0, frame.annotations)
#     add_annotations_to_camera_ax(ax[0, 1], frame.cameras.cam_f0, frame.annotations)
#     add_annotations_to_camera_ax(ax[0, 2], frame.cameras.cam_r0, frame.annotations)

#     add_annotations_to_camera_ax(ax[1, 0], frame.cameras.cam_l1, frame.annotations)
#     add_configured_bev_on_ax(ax[1, 1], scene.map_api, frame)
#     add_annotations_to_camera_ax(ax[1, 2], frame.cameras.cam_r1, frame.annotations)

#     add_annotations_to_camera_ax(ax[2, 0], frame.cameras.cam_l2, frame.annotations)
#     add_annotations_to_camera_ax(ax[2, 1], frame.cameras.cam_b0, frame.annotations)
#     add_annotations_to_camera_ax(ax[2, 2], frame.cameras.cam_r2, frame.annotations)

#     configure_all_ax(ax)
#     configure_bev_ax(ax[1, 1])
#     fig.tight_layout()
#     fig.subplots_adjust(wspace=0.01, hspace=0.01, left=0.01, right=0.99, top=0.99, bottom=0.01)

#     return fig, ax

def plot_cameras_frame_with_annotations(scene: Scene, frame_idx: int) -> Tuple[plt.Figure, Any]:
    """
    Plots only the cam_f0 camera image with annotations.
    :param scene: navsim scene dataclass
    :param frame_idx: index of selected frame
    :return: figure and ax object of matplotlib
    """
    frame = scene.frames[frame_idx]

    # 创建单一子图
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))  # 可以根据实际需求调整figsize

    # 添加cam_f0的图像和标注
    add_annotations_to_camera_ax(ax, frame.cameras.cam_f0, frame.annotations)

    # 可选：配置坐标轴（去掉刻度、边框等）
    ax.axis("off")

    fig.tight_layout()
    
    return fig, ax

def frame_plot_to_pil(
    callable_frame_plot: Callable[[Scene, int], Tuple[plt.Figure, Any]],
    scene: Scene,
    frame_indices: List[int],
) -> List[Image.Image]:
    """
    Plots a frame according to plotting function and return a list of PIL images
    :param callable_frame_plot: callable to plot a single frame
    :param scene: navsim scene dataclass
    :param frame_indices: list of indices to save
    :return: list of PIL images
    """

    images: List[Image.Image] = []

    for frame_idx in tqdm(frame_indices, desc="Rendering frames"):
        fig, ax = callable_frame_plot(scene, frame_idx)

        # Creating PIL image from fig
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        images.append(Image.open(buf).copy())

        # close buffer and figure
        buf.close()
        plt.close(fig)

    return images


def frame_plot_to_gif(
    file_name: str,
    callable_frame_plot: Callable[[Scene, int], Tuple[plt.Figure, Any]],
    scene: Scene,
    frame_indices: List[int],
    duration: float = 500,
) -> None:
    """
    Saves a frame-wise plotting function as GIF (hard G)
    :param callable_frame_plot: callable to plot a single frame
    :param scene: navsim scene dataclass
    :param frame_indices: list of indices
    :param file_name: file path for saving to save
    :param duration: frame interval in ms, defaults to 500
    """
    images = frame_plot_to_pil(callable_frame_plot, scene, frame_indices)
    images[0].save(file_name, save_all=True, append_images=images[1:], duration=duration, loop=0)
