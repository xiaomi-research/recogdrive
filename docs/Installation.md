Installation for ReCogDrive
# Installation for ReCogDrive

After successfully downloading the NAVSIM dataset:

## 1. Pretraining (SFT on VLM)
If you want to perform **ReCogDrive pretraining** and do **SFT training on VLM**, install InternVL dependencies:
```bash
pip install -r internvl_chat/internvl_chat.txt
```

##  2. Training & Evaluation on NAVSIM
If you want to train and evaluate ReCogDrive on NAVSIM, install the requirements:

```bash
cd /path/to/ReCogDrive
pip install -e .
# NAVSIM 2.0 uses the same shared ReCogDrive package:
NAVSIM_TREE=2.0 pip install -e .
# Optional LoRA and VQA benchmark runtime:
pip install -e '.[lora,vqa]'
```
