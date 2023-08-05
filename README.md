# Joint Skeletal and Semantic Embedding Loss for Micro-gesture Classification

The solution of HFUT-VUT Team for the [The 1st Workshop & Challenge on Micro-gesture Analysis for Hidden Emotion Understanding (MiGA)](https://cv-ac.github.io/MiGA2023/), please refer to the [arxiv paper](https://arxiv.org/abs/2307.10624) for more details. 

## Installation
```bash
git clone https://github.com/VUT-HFUT/MiGA2023_Track1.git
# This command runs well with conda 22.9.0, if you are running an early conda version and got some errors, try to update your conda first
conda env create -f pyskl.yaml
conda activate pyskl
pip install -e .
```

## Data preparation
1. Please first download the skeleton modality of the iMiGUE dataset via Codalab platform by participating the [MiGA competition](https://codalab.lisn.upsaclay.fr/competitions/11758). 
2. Convert the raw skeleton data to the [PYSKL data format](https://github.com/kennymckormick/pyskl/blob/main/tools/data/README.md). You can refer to `./tools/readlabel.py`. 
3. By the way, we separate the validation set from the training set. 

## Training & Testing & Ensemble
You can use following commands for training and testing. 
```bash
# training
## Note that this process will consume 8 hours with two NVIDIA 3090 for each model.
## joint model
bash tools/dist_train.sh ./configs/posec3d/slowonly_r50_imigue_2dkp_emb20/joint.py 2 
## limb model
bash tools/dist_train.sh ./configs/posec3d/slowonly_r50_imigue_2dkp_emb20/limb.py 2 

# test
## joint model
python tools/test.py ./configs/posec3d/slowonly_r50_imigue_2dkp_emb20/joint.py -C ./work_dirs/posec3d/weight/slowonly_r50_imigue_2dkp_emb20/joint/epoch100/emb_20/best_top1_acc_epoch_85.pth

## limb model
python tools/test.py ./configs/posec3d/slowonly_r50_imigue_2dkp_emb20/limb.py -C ./work_dirs/posec3d/weight/slowonly_r50_imigue_2dkp_emb20/limb/e100/emb20/best_top1_acc_epoch_86.pth

# ensemble
## We ensemble joint and limb model with weighted sum.
cd ensemble
python ensemble.py
```
The generated `Sumission.zip` is the final result.

## Citation
If you use this code in your research, please consider citing:
```
@article{li2023joint,
  title={Joint Skeletal and Semantic Embedding Loss for Micro-gesture Classification},
  author={Li, Kun and Guo, Dan and Chen, Guoliang and Peng, Xinge and Wang, Meng},
  journal={arXiv preprint arXiv:2307.10624},
  year={2023}
}
```
## Citation
This code began with [PYSKL toolbox](https://github.com/kennymckormick/pyskl/tree/main). We thank the developers for doing most of the heavy-lifting.

## Contact
For any questoions, feel free to contact: kunli.hfut@gmail.com