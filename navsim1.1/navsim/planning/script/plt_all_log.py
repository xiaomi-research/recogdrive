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
from typing import List, Dict

import torch
import torch.distributed as dist

from navsim.common.dataloader import SceneLoader, MetricCacheLoader # SceneLoader 
from navsim.common.dataclasses import SceneFilter, SensorConfig, Trajectory
from navsim.visualization.plots import plot_bev_and_camera_with_agent
from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
import pandas as pd

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
openscene_data_root = Path(os.getenv("OPENSCENE_DATA_ROOT", "."))

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

# SceneLoader 和 MetricCacheLoader 初始化 (保持不变)
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

simulator = PDMSimulator(TrajectorySampling(time_horizon=4, interval_length=0.1))
scorer = PDMScorer(TrajectorySampling(time_horizon=4, interval_length=0.1), PDMScorerConfig())

# --- 核心辅助函数 ---

def find_one_score_tokens(file_path: str) -> List[Dict[str, any]]:
    """读取CSV文件，提取score为1的token条目 (保持不变)"""
    try:
        df = pd.read_csv(file_path)
        # 筛选 score 为 1 的条目
        zero_score_df = df[df['score'] == 1] 
        
        token_entries = []
        for index, row in zero_score_df.iterrows():
            entry = {'token': row['token']}
            for key in METRIC_KEYS:
                if key in row:
                     # 确保保存的数值是浮点数
                    entry[key] = float(row[key]) if key != 'token' else row[key]
            token_entries.append(entry)
        return token_entries

    except FileNotFoundError:
        print("CSV文件未找到，请检查文件路径。")
        return []
    except KeyError as e:
        print(f"CSV文件缺少必要的列：{e}。")
        return []
    except Exception as e:
        print(f"读取文件时出错: {str(e)}")
        return []

def save_low_frequency_video_frames(log_name: str, sorted_tokens: List[str], agent, scene_loader, scene_loader_traj, metric_cache_loader):
    """
    处理特定 log 下所有采样场景（token）的中心帧，并将结果图片按顺序保存，生成一个低频连续视频的序列。
    """
    
    # 构造 log 对应的输出目录
    output_dir = os.path.join("traj_plot_vis_test_navsim_low_freq_video_IL", log_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历该 Log 下所有按顺序排列的采样中心帧 Token
    for frame_idx, token in enumerate(tqdm(sorted_tokens, desc=f"Rank {rank} | Processing Log: {log_name}")):
        
        # 1. 加载 Scene/AgentInput (使用修正后的 get_scene_from_token，它使用采样数据)
        try:
            scene = scene_loader.get_scene_from_token(token)
            scene_traj = scene_loader_traj.get_scene_from_token(token) 
        except KeyError:
            # 如果 Token 不是采样中心帧，会被 SceneLoader 抛出 KeyError。跳过。
            print(f"Warning: Token {token} is not a sampled center frame. Skipping.")
            continue
        except Exception as e:
            print(f"Warning: Failed to load scene for token {token} in log {log_name}: {e}. Skipping frame.")
            continue

        # current_frame_idx：短场景（Scene）中的当前帧索引 (num_history_frames - 1)
        current_frame_idx = scene_loader._scene_filter.num_history_frames - 1
        
        # --- Metric Cache 加载 (可选) ---
        metric_cache_path = metric_cache_loader.metric_cache_paths.get(token)
        if metric_cache_path:
            try:
                with lzma.open(metric_cache_path, "rb") as f:
                    pickle.load(f)
            except: 
                pass 
        
        # 2. 生成可视化图
        fig, _, _ = plot_bev_and_camera_with_agent(scene, scene_traj, current_frame_idx, agent)
        
        # 3. 保存图片
        plt.savefig(
            os.path.join(output_dir, f"{log_name}_{frame_idx:05d}_{token}.png"), 
            bbox_inches='tight',
            dpi=200
        )
        plt.close(fig)
        
    print(f"Rank {rank}: Log {log_name} 已处理 {len(sorted_tokens)} 个连续视频帧，保存至 {output_dir}")

# --- 主执行逻辑 ---

if __name__ == '__main__':
    
    # 1. 映射：将采样的中心帧 Token 归类到其 Log Name 下，并保留原始帧数据
    log_to_sampled_data = {}
    
    # 遍历所有采样的短场景（键是中心帧 Token）
    for token, scene_frame_list in scene_loader.scene_frames_dicts.items():
        # 获取采样场景的中心帧字典
        center_frame_idx = scene_loader._scene_filter.num_history_frames - 1
        center_frame_dict = scene_frame_list[center_frame_idx]
        
        # 获取 Log Name
        try:
            # 使用包含所有 Token 的映射来获取 Log Name
            log_name = scene_loader.token_to_log_file[token] 
        except KeyError:
            print(f"Error: Sampled token {token} not mapped to any log. Skipping.")
            continue
            
        if log_name not in log_to_sampled_data:
            log_to_sampled_data[log_name] = []
            
        # 存储中心帧的帧字典，用于后续排序
        log_to_sampled_data[log_name].append(center_frame_dict)

    all_logs = list(log_to_sampled_data.keys())
    
    if not all_logs:
        print("未找到任何采样的 Log 文件。")
    else:
        # 2. 分布式划分 Log 文件
        local_logs = all_logs[rank::world_size]
        
        print(f"Rank {rank} 分配到 {len(local_logs)} 个 Log 文件进行处理。")

        # 3. 遍历并处理分配给本进程的所有 Log 文件
        for log_name in local_logs:
            sampled_frames_for_log = log_to_sampled_data[log_name]
            
            # --- 关键步骤：按时间戳（timestamp）排序 ---
            sampled_frames_for_log.sort(key=lambda x: x['timestamp'])
            
            # 提取排序后的 Token 列表
            sorted_tokens = [frame_dict['token'] for frame_dict in sampled_frames_for_log]
            
            # 调用新的函数处理按时间顺序排列的中心帧 Token
            save_low_frequency_video_frames(
                log_name, 
                sorted_tokens, 
                agent, 
                scene_loader, 
                scene_loader_traj, 
                metric_cache_loader
            )

    # 确保所有进程完成后再退出
    if world_size > 1:
        dist.barrier()
        
    print(f"Rank {rank} 所有任务完成。")