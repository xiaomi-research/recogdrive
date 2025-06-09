

<div align="center">
<h3>ReCogDrive: A Reinforced Cognitive Framework for End-to-End Autonomous Driving</h3>

Yongkang Li<sup>1,2\*</sup>, Kaixin Xiong<sup>2\*</sup>, Xiangyu Guo<sup>1,2</sup>, Fang Li<sup>2</sup>, Sixu Yan<sup>1</sup>, Gangwei Xu<sup>1,2</sup>,  
Lijun Zhou<sup>2</sup>, Long Chen<sup>2</sup>, Haiyang Sun<sup>2†</sup>, Bing Wang<sup>2</sup>, Guang Chen<sup>2</sup>,  
Hangjun Ye<sup>2</sup>, Wenyu Liu<sup>1</sup>, Xinggang Wang<sup>1✉</sup>  

<sup>1</sup>Huazhong University of Science and Technology  
<sup>2</sup>Xiaomi EV  

(\*) Equal contribution. (†) Project leader. (✉) Corresponding author.  

<a href=""><img src='https://img.shields.io/badge/arXiv-ReCogDrive-red' alt='Paper PDF'></a>
<a href="https://xiaomi-research.github.io/recogdrive/"><img src='https://img.shields.io/badge/Project_Page-ReCogDrive-green' alt='Project Page'></a>
</div>


<!-- ## Introduction -->
## Abstract
Although end-to-end autonomous driving has made remarkable progress, its performance degrades significantly in rare and long-tail scenarios. Recent approaches attempt to address this challenge by leveraging the rich world knowledge of Vision-Language Models (VLMs), but these methods suffer from several limitations: 
(1) a significant domain gap between the pre-training data of VLMs and real-world driving data, 
(2) a dimensionality mismatch between the discrete language space and the continuous action space, 
(3) imitation learning tends to capture the average behavior present in the dataset, which may be suboptimal even dangerous.
In this paper, we propose CogDrive, an autonomous driving system that integrates VLMs with a diffusion planner, adopting a three-stage paradigm for training. In the first stage, we use a large-scale driving question-answering dataset to train the VLMs, mitigating the domain discrepancy between generic content and real-world driving scenarios.
In the second stage, we employ a diffusion-based planner to perform imitation learning, mapping representations from the latent language space to continuous driving actions. Finally, we fine-tune the diffusion planner using reinforcement learning in the NAVSIM non-reactive simulator, enabling the model to generate safer, more human-like driving trajectories.
We evaluate our approach on the planning-oriented NAVSIM benchmark, achieving a PDMS of 89.6 and setting a new state-of-the-art that surpasses the previous vision-only SOTA by 5.6 PDMS.
         

## Overview
<div align="center">
<img src="assets/images/framework.jpg" width="1000">
</div>

<!-- ## News

