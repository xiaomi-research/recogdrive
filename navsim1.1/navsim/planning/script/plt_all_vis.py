import os
import json
from pathlib import Path
import csv
import hydra
from hydra.utils import instantiate
import matplotlib.pyplot as plt
import pickle
import lzma
from tqdm import tqdm

import torch
import torch.distributed as dist

from navsim.common.dataloader import SceneLoader, MetricCacheLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig, Trajectory
from navsim.visualization.plots import (
    plot_cameras_frame,
    plot_cameras_frame_with_agent,
    plot_traj_with_agent,
    plot_bev_and_camera_with_agent,
    plot_bev_with_agent
)
from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
from navsim.evaluate.pdm_score import pdm_score
import pandas as pd

# 分布式初始化函数
def init_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        print(f"Distributed init: rank {rank}/{world_size}, local_rank {local_rank}")
        return rank, world_size, local_rank
    else:
        return 0, 1, 0

rank, world_size, local_rank = init_distributed()
device = torch.device("cuda", local_rank)

SPLIT = "test"  # ["mini", "test", "trainval"]
FILTER = "navtest"

hydra.initialize(config_path="./navsim/planning/script/config/common/train_test_split/scene_filter")
cfg = hydra.compose(config_name=FILTER)
scene_filter: SceneFilter = instantiate(cfg)
openscene_data_root = Path(os.getenv("OPENSCENE_DATA_ROOT"))

# 初始化 agent，并将其放置到对应 GPU 上
agent = ReCogDriveAgent(
    TrajectorySampling(time_horizon=4, interval_length=0.5),
    checkpoint_path='',
    vlm_path='',
    cam_type='single',
    vlm_type='internvl',
    dit_type='small',
    sampling_method='ddim',
    cache_mode='False',
    cache_hidden_state='False',
    vlm_size='small',
    grpo=False,
).to(device)

agent.initialize()
# 如果需要分布式封装，则取消注释下面代码
# if world_size > 1:
#     agent = torch.nn.parallel.DistributedDataParallel(agent, device_ids=[local_rank])
#     print("Agent wrapped with DistributedDataParallel")

def find_one_score_tokens(file_path):
    """读取CSV文件，提取score为0的token条目"""
    try:
        df = pd.read_csv(file_path)
        zero_score_df = df[df['score'] == 1]
        tokens = zero_score_df['token'].tolist()
        token_entries = []
        for token in tokens:
            row = df[df['token'] == token].iloc[0]
            entry = {'token': token}
            for key in METRIC_KEYS:
                # 确保保存的数值是浮点数，而不是格式化后的字符串
                entry[key] = float(row[key]) if key != 'token' else row[key]
            token_entries.append(entry)
        return token_entries

    except FileNotFoundError:
        print("CSV文件未找到，请检查文件路径。")
        return []
    except KeyError:
        print("CSV文件缺少必要的列（token或score）。")
        return []
    except Exception as e:
        print(f"读取文件时出错: {str(e)}")
        return []

scene_loader = SceneLoader(
    openscene_data_root / f"navsim_logs/{SPLIT}",
    openscene_data_root / f"sensor_blobs/{SPLIT}",
    scene_filter,
    sensor_config=agent.module.get_sensor_config() if hasattr(agent, "module") else agent.get_sensor_config(),
)
scene_loader_traj = SceneLoader(
    openscene_data_root / f"navsim_logs/{SPLIT}",
    openscene_data_root / f"sensor_blobs/{SPLIT}",
    scene_filter,
    sensor_config=agent.module.get_sensor_config() if hasattr(agent, "module") else agent.get_sensor_config(),
    load_image_path=True
)

METRIC_KEYS = [
    'no_at_fault_collisions',
    'drivable_area_compliance',
    'ego_progress',
    'time_to_collision_within_bound',
    'comfort',
    'driving_direction_compliance',
    'score'
]

metric_cache_loader = MetricCacheLoader(Path(''))

proposal_sampling = TrajectorySampling(time_horizon=4, interval_length=0.1)

simulator = PDMSimulator(proposal_sampling)

scorer = PDMScorer(proposal_sampling, PDMScorerConfig())


def save_perfect_samples(entries, agent, metric_cache_loader, simulator, scorer):
    output_dir = "traj_plot_vis_test_navsim_perfect_rl_new"
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用 tqdm 显示进度条，描述中显示当前进程 rank 信息
    for entry in tqdm(entries, desc=f"Rank {rank} processing tokens"):
        token = entry['token']
        scene = scene_loader.get_scene_from_token(token)
        scene_traj = scene_loader_traj.get_scene_from_token(token)
        frame_idx = scene.scene_metadata.num_history_frames - 1

        metric_cache_path = metric_cache_loader.metric_cache_paths[token]

        with lzma.open(metric_cache_path, "rb") as f:
            metric_cache = pickle.load(f)

        # 生成带 agent 轨迹的复合图，并获取 pdm_result 和 agent_trajectory
        fig, _, _ = plot_bev_and_camera_with_agent(scene, scene_traj, frame_idx, agent)
        plt.savefig(
            os.path.join(output_dir, f"{entry['token']}_vis_traj.png"),
            bbox_inches='tight',
            dpi=200
        )
        plt.close(fig)  # close figure to prevent memory leak
        

def create_entries_from_tokens(tokens):
    """根据 token 列表创建条目"""
    token_entries = []
    for token in tokens:
        entry = {'token': token}
        token_entries.append(entry)
    return token_entries


# all_tokens = scene_loader.tokens  # 从 scene_loader 获取所有 token
# all_tokens_entries = create_entries_from_tokens(all_tokens)  # 根据 tokens 创建条目


file_path = ''  # Replace with your actual filename
one_score_entries = find_one_score_tokens(file_path)


if one_score_entries:
    # 按照 rank 划分 token，每个进程只处理部分 token
    local_entries = one_score_entries[rank::world_size]
    save_perfect_samples(local_entries, agent, metric_cache_loader, simulator, scorer)
    print(f"Rank {rank}: 已保存 {len(local_entries)} 个样本")
else:
    print("未找到符合条件的样本")
