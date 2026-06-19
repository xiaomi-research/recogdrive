<div align="center">
<img src="assets/images/logo2.png" width="200">
<h1>ReCogDrive</h1>
<h3>A Reinforced Cognitive Framework for End-to-End Autonomous Driving</h3>

[Yongkang Li](https://owl-10.github.io/yongkangli/)<sup>1,2\*</sup>, Kaixin Xiong<sup>2\*</sup>, Xiangyu Guo<sup>1,2</sup>, Fang Li<sup>2</sup>, [Sixu Yan](https://sixu-yan.github.io/)<sup>1</sup>, [Gangwei Xu](https://gangweix.github.io/)<sup>1,2</sup>,  
Lijun Zhou<sup>2</sup>, [Long Chen](https://long.ooo/)<sup>2</sup>, Haiyang Sun<sup>2†</sup>, Bing Wang<sup>2</sup>, Kun Ma<sup>2</sup>, Guang Chen<sup>2</sup>,  
Hangjun Ye<sup>2</sup>, [Wenyu Liu](https://eic.hust.edu.cn/professor/liuwenyu/)<sup>1</sup>, [Xinggang Wang](https://xwcv.github.io/)<sup>1✉</sup>  

<sup>1</sup>Huazhong University of Science and Technology  
<sup>2</sup>Xiaomi EV  

(\*) Equal contribution. (†) Project leader. (✉) Corresponding author.  

International Conference on Learning Representation (ICLR), 2026

<a href="https://arxiv.org/abs/2506.08052"><img src='https://img.shields.io/badge/arXiv-ReCogDrive-red' alt='Paper PDF'></a>   <a href="https://xiaomi-research.github.io/recogdrive/"><img src='https://img.shields.io/badge/Project_Page-ReCogDrive-green' alt='Project Page'></a> [![huggingface collection](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-Recogdrive-yellow)](https://huggingface.co/collections/owl10/recogdrive-68bafa143de172bab8de5752)&nbsp; [![huggingface datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Datasets-Recogdrive-red)](https://huggingface.co/datasets/owl10/ReCogDrive_Pretraining)&nbsp;
</div>


</div>

## News
* **` Apr. 2nd, 2026`:** Please also follow our latest work [UniDriveVLA](https://github.com/xiaomi-research/unidrivevla), where we continue exploring Vision-Language-Action (VLA) models for autonomous driving. UniDriveVLA extends our research direction toward unified driving understanding, reasoning, and action generation. Stay tuned for more updates!
* **` Jan. 26th, 2025`:** Our paper has been accepted at ICLR 2026. 🎉
* **` Sept. 30th, 2025`:** We have updated our latest paper with more model details, experiments, and comprehensive visualizations. Meanwhile, we fixed the unintended NumPy issue 🐛 that previously caused inconsistencies in the training metric cache. Now the code ensures reproducible and consistent results. Special thanks to the discussion in [issue #10](https://github.com/riron1206) for bringing this up!
* **` Aug. 24th, 2025`:** We have released all driving pretraining QA, including 12 driving datasets and our own annotated NavSim data. We have rewritten the scoring, filtering, and evaluation for open-source data. If it’s helpful to you, feel free to star and cite our work! 🚗💨
* **` Aug. 21th, 2025`:** We release the initial version of code and weight on NAVSIM, along with documentation and training/evaluation scripts. We will also update our new revision of the paper and the pretraining datasets later this month or next month. Please stay tuned! ☕️
* **` Jun. 11th, 2025`:** We released our paper on [Arxiv](https://arxiv.org/abs/2506.08052). Code/Models are coming soon. Please stay tuned! ☕️


## Updates
- [ ] Release Bench2Drive, DriveLM, NAVSIM2.0, Drivebench evaluation frameworks
- [x] Release Paper
- [x] Release Full Models and Training/Evaluation Framework
- [x] Release Full Driving QA Datasets
- [x] Release Updated Paper

## Table of Contents
- [News](#news)
- [Updates](#updates)
- [Table of Contents](#table-of-contents)
- [Abstract](#abstract)
- [Getting Started](#getting-started)
- [Checkpoint](#checkpoint)
- [Driving Pretraining Datasets](#driving-pretraining-datasets)
- [Qualitative Results on NAVSIM Navtest](#qualitative-results-on-navsim-navtest)
- [Qualitative Results on Bench2drive](#qualitative-results-on-bench2drive)
- [Contact](#contact)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)

## Abstract         

Recent studies have explored leveraging the world knowledge and cognitive capabilities of Vision-Language Models (VLMs) to address the long-tail problem in end-to-end autonomous driving. However, existing methods typically formulate trajectory planning as a language modeling task, where physical actions are output in the language space, potentially leading to issues such as format-violating outputs, infeasible actions, and slow inference speeds. In this paper, we propose ReCogDrive, a novel **Re**inforced **Cog**nitive framework for end-to-end autonomous **Driv**ing, unifying driving understanding and planning by integrating an autoregressive model with a diffusion planner. First, to instill human driving cognition into the VLM, we introduce a hierarchical data pipeline that mimics the sequential cognitive process of human drivers through three stages: generation, refinement, and quality control. Building on this cognitive foundation, we then address the language-action mismatch by injecting the VLM's learned driving priors into a diffusion planner to efficiently generate continuous and stable trajectories. Furthermore, to enhance driving safety and reduce collisions, we introduce a Diffusion Group Relative Policy Optimization (DiffGRPO) stage, reinforcing the planner for enhanced safety and comfort. Extensive experiments on the NAVSIM and Bench2Drive benchmarks demonstrate that ReCogDrive achieves state-of-the-art performance. Additionally, qualitative results across diverse driving scenarios and DriveBench highlight the model's scene comprehension. Code and models are available at [ReCogDrive GitHub Repository](https://github.com/xiaomi-research/recogdrive).

<div align="center">
<img src="assets/images/framework.png" width="1000">
</div>



## Getting Started

- [Download NAVSIM datasets following official instruction](https://github.com/autonomousvision/navsim/blob/main/docs/install.md)
- [Preparation of ReCogDrive environment](docs/Installation.md)
- [ReCogDrive Training and Evaluation](docs/Train_Eval.md)

## Checkpoint

> Results on NAVSIM


| Method | Model Size | Training Stage | PDMS | Weight Download |
| :---: | :---: | :---: | :---: |  :---: |
| ReCogDrive-Base-VLM | 2B | Stage 1 | 84.1 | [Model](https://huggingface.co/owl10/ReCogDrive-VLM-2B/tree/main) | |
| ReCogDrive-Base-IL | 2B + 35M | Stage 1&2| 86.5 | [Model](https://huggingface.co/owl10/ReCogDrive-2B-IL/tree/main) | |
| ReCogDrive-Base-RL | 2B + 35M | Stage 1&2&3| 90.8 | [Model](https://huggingface.co/owl10/ReCogDrive-2B-RL/tree/main) | |
| ReCogDrive-Large-VLM | 8B | Stage 1 | 86.4 | [Model](https://huggingface.co/owl10/ReCogDrive-VLM-8B/tree/main) | |
| ReCogDrive-Large-IL | 8B + 35M | Stage 1&2| 86.5 | [Model](https://huggingface.co/owl10/ReCogDrive-8B-IL/tree/main) | |
| ReCogDrive-Large-RL | 8B + 35M | Stage 1&2&3| 90.4 | [Model](https://huggingface.co/owl10/ReCogDrive-8B-RL/tree/main) | |


> Results on Bench2drive

<table style="border-collapse: collapse; text-align: center; width: 100%;">
  <caption><b>Closed-loop and Multi-ability Testing Results in CARLA Bench2Drive Leaderboard</b></caption>
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th colspan="4">Closed-loop Metric ↑</th>
      <th colspan="5">Multi-Ability Test (%) ↑</th>
    </tr>
    <tr>
      <th>Efficiency</th>
      <th>Comfort</th>
      <th>Success</th>
      <th>DS</th>
      <th>Merging</th>
      <th>Overtaking</th>
      <th>Emerg. Brake</th>
      <th>GiveWay</th>
      <th>Traf. Sign</th>
      <th>Mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>ReCogDrive</b></td>
      <td>138.18</td>
      <td>17.45</td>
      <td><b>45.45</b></td>
      <td><b>71.36</b></td>
      <td><b>29.73</b></td>
      <td>20.00</td>
      <td><b>69.09</b></td>
      <td>20.00</td>
      <td><b>71.34</b></td>
      <td>42.03</td>
    </tr>
  </tbody>
</table>



> Results on DriveLM and DriveBench
<table>
  <thead>
    <tr>
      <th rowspan="2" style="text-align:center">Method</th>
      <th rowspan="2" style="text-align:center">DriveLM (GPT-Score)</th>
      <th rowspan="2" style="text-align:center">LingoQA (Lingo-Judge)</th>
      <th colspan="5" style="text-align:center">DriveBench</th>
    </tr>
    <tr>
      <th style="text-align:center">Percep.</th>
      <th style="text-align:center">Predict.</th>
      <th style="text-align:center">Plan.</th>
      <th style="text-align:center">Behav.</th>
      <th style="text-align:center">Avg.</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align:center">
        <a href="https://huggingface.co/owl10/ReCogDrive-VLM-DriveLM/tree/main" target="_blank">ReCogDrive</a>
      </td>
      <td style="text-align:center"><strong>67.30</strong></td>
      <td style="text-align:center"><strong>67.20</strong></td>
      <td style="text-align:center">64.95</td>
      <td style="text-align:center">49.34</td>
      <td style="text-align:center">70.20</td>
      <td style="text-align:center">42.36</td>
      <td style="text-align:center"><strong>56.71</strong></td>
    </tr>
  </tbody>
</table>



## Driving Pretraining Datasets
| Datasets | Source |  Rewritten and filtered Annotations Jsonl |
| :---: | :---: | :---: |
| NAVSIM-Traj | - | [JSONL](https://huggingface.co/datasets/owl10/ReCogDrive_Pretraining/tree/main/Navsim_Traj)  |
| NAVSIM-ReCogDrive | - |  [JSONL](https://huggingface.co/datasets/owl10/ReCogDrive_Pretraining/tree/main/Navsim_ReCogDrive)   |
| DriveLM | [link](https://github.com/OpenDriveLab/DriveLM) | [JSONL](https://huggingface.co/datasets/owl10/ReCogDrive_Pretraining/tree/main/DriveLM) |
| Nuinstruct | [link](https://github.com/xmed-lab/NuInstruct) |  [JSONL](https://huggingface.co/datasets/owl10/ReCogDrive_Pretraining/tree/main/Nuinstruct)  |
| NuscenesQA | [link](https://github.com/qiantianwen/NuScenes-QA) | [JSONL](https://huggingface.co/datasets/owl10/ReCogDrive_Pretraining/tree/main/Nuscenes-QA) |
| Omnidrive | [link](https://github.com/NVlabs/OmniDrive) |  [JSONL](https://huggingface.co/datasets/owl10/ReCogDrive_Pretraining/tree/main/Omnidrive)  |
| Senna | [link](https://github.com/hustvl/Senna) |  [JSONL](https://huggingface.co/datasets/owl10/ReCogDrive_Pretraining/tree/main/Senna)   |
| LingoQA | [link](https://github.com/wayveai/LingoQA) |  [JSONL](https://huggingface.co/datasets/owl10/ReCogDrive_Pretraining/tree/main/LingoQA)  |
| Drama | [link](https://usa.honda-ri.com/drama)  |  [JSONL](https://huggingface.co/datasets/owl10/ReCogDrive_Pretraining/tree/main/Drama)  |
| MapLM | [link](https://github.com/LLVM-AD/MAPLM)  |  [JSONL](https://huggingface.co/datasets/owl10/ReCogDrive_Pretraining/tree/main/Maplm)  |
| Talk2Car | [link](https://github.com/talk2car/Talk2Car)  |  [JSONL](https://huggingface.co/datasets/owl10/ReCogDrive_Pretraining/tree/main/Talk2Car)  |
| Drivegpt4 | [link](https://tonyxuqaq.github.io/projects/DriveGPT4/)  |  [JSONL](https://huggingface.co/datasets/owl10/ReCogDrive_Pretraining/tree/main/Drivegpt4)  |
| CODA-LM | [link](https://coda-dataset.github.io/coda-lm/)  |  [JSONL](https://huggingface.co/datasets/owl10/ReCogDrive_Pretraining/tree/main/CODA-LM)  |
| SUTD | [link](https://github.com/SUTDCV/SUTD-TrafficQA)  |  [JSONL](https://huggingface.co/datasets/owl10/ReCogDrive_Pretraining/tree/main/SUTD)  |
| Bench2drive-Traj | - | [JSONL](https://huggingface.co/datasets/owl10/ReCogDrive_Pretraining/tree/main/Bench2drive_Traj)  |
| Bench2drive-QA | [link](https://github.com/Thinklab-SJTU/Bench2Drive-VL) |  [JSONL](https://huggingface.co/datasets/owl10/ReCogDrive_Pretraining/tree/main/Bench2drive_QA)   |

Our ReCogDrive is pretrained on 12 open-source driving datasets. For most of these datasets, we leveraged Qwen2.5VL-72B to re-annotate the answers, applied standardized scoring, and filtered them to obtain 12 high-quality QA datasets. In addition, we built an automated annotation pipeline on Navsim, generating 752k QA pairs. These resources enable VLMs to better adapt to driving scenarios. **If you only want to train a VLM for planning on a specific dataset, you can use just that dataset’s trajectories and QA (for example, NAVSIM-Traj and NAVSIM-ReCogDrive) to train the VLM and then perform planning; this can achieve results close to training on the full dataset. We perform large-scale pretraining to improve the VLM’s understanding across diverse driving scenarios.**

We open-sourced these high-quality driving QA datasets in the hope of supporting research on Vision-Language-Action (VLA) for driving. If the official maintainers of any dataset prefer that we do not release the JSON annotations, we will remove them immediately. Please note that if you use these datasets, you must comply with the original licenses of the respective datasets. We emphasize that our usage of these datasets is solely for academic research purposes, with no commercial applications involved.

In addition, we provide training data on Bench2Drive, where we further fine-tune our models on mixed data and Navsim real-world scenarios, followed by training on Bench2Drive-Traj and Bench2Drive-QA to better adapt to the CARLA driving environment.



## Qualitative Results on NAVSIM Navtest 
<div align="center">
  <img src="assets/images/vis.png" width="1000">
</div>
<p align="left">
  We compare ReCogDrive (IL and RL) with Transfuser, where RL yields safer and more reliable trajectories in challenging turning scenarios. More visualizations are in the supplementary material.
</p>



## Qualitative Results on Bench2drive
<div align="center">
  <img src="assets/images/b2d.png" width="1000">
</div>
<div align="center">
  <img src="assets/images/b2d2.png" width="1000">
</div>
<p align="left">
  This visualization demonstrates the driving capabilities of <b>ReCogDrive</b> across diverse scenarios in both real-world settings and the CARLA-simulated Bench2Drive environment. The results show that our model can handle complex maneuvers such as lane following, turning, and interacting with traffic signs, reflecting strong adaptability to various driving contexts.

</p>



## Contact
If you have any questions, please contact [Yongkang Li](https://owl-10.github.io/yongkangli/) via email (liyk@hust.edu.cn) or wechat (liyk_0803).

## Acknowledgement
ReCogDrive is greatly inspired by the following outstanding contributions to the open-source community: [NAVSIM](https://github.com/autonomousvision/navsim), [DPPO](https://github.com/irom-princeton/dppo), [LightningDiT](https://github.com/hustvl/LightningDiT), [DiffusionDrive](https://github.com/hustvl/DiffusionDrive), [Senna](https://github.com/hustvl/Senna), [GR00T](https://github.com/NVIDIA/Isaac-GR00T).


## Citation
If you find ReCogDrive is useful in your research or applications, please consider giving us a star 🌟 and citing it by the following BibTeX entry.

```bibtex
@article{li2025recogdrive,
  title={ReCogDrive: A Reinforced Cognitive Framework for End-to-End Autonomous Driving},
  author={Li, Yongkang and Xiong, Kaixin and Guo, Xiangyu and Li, Fang and Yan, Sixu and Xu, Gangwei and Zhou, Lijun and Chen, Long and Sun, Haiyang and Wang, Bing and others},
  journal={arXiv preprint arXiv:2506.08052},
  year={2025}
}
```

