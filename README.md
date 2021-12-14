# SphereRPN

Code for the paper **SphereRPN: Learning Spheres for High-Quality Region Proposals on 3D Point Clouds Object Detection**, ICIP 2021.

**Authors**: Thang Vu, Kookhoi Kim, Haeyong Kang, Xuan Thanh Nguyen, Tung M. Luu, Chang D. Yoo

This work was partly supported by Institute for Information & communications Technology Planning & Evaluation(IITP) grant funded by the Korea government(MSIT) (2021-0-01381, Development of Causal AI through Video Understanding and Reinforcement Learning, and Its Applications to Real Environments), and partly supported by Institute for Information & communications Technology Planning & Evaluation(IITP) grant funded by the Korea government(MSIT) (No. 2019-0-01396, Development of framework for analyzing, detecting, mitigating of bias in AI model and training data).


## Installation

### Requirements
* Python 3.7.0
* Pytorch 1.1.0
* CUDA 9.0

### Virtual Environment
```
conda create -n sphere_rpn python==3.7
source activate sphere_rpn
```

### Install

(1) Clone the repository.
```
git clone https://github.com/llijiang/PointGroup.git --recursive 
cd PointGroup
```

(2) Install the dependent libraries.
```
pip install -r requirements.txt
conda install -c bioconda google-sparsehash 
```

(3) For the SparseConv, we apply the implementation of [spconv](https://github.com/traveller59/spconv). The repository is recursively downloaded at step (1). We use the version 1.0 of spconv. 

**Note:** We further modify `spconv\spconv\functional.py` to make `grad_output` contiguous. Make sure you use our modified `spconv`.

* To compile `spconv`, firstly install the dependent libraries. 
```
conda install libboost
conda install -c daleydeng gcc-5 # need gcc-5.4 for sparseconv
```
Add the `$INCLUDE_PATH$` that contains `boost` in `lib/spconv/CMakeLists.txt`. (Not necessary if it could be found.)
```
include_directories($INCLUDE_PATH$)
```

* Compile the `spconv` library.
```
cd lib/spconv
python setup.py bdist_wheel
```

* Run `cd dist` and use pip to install the generated `.whl` file.



(4) Compile the `pointgroup_ops` library.
```
cd lib/pointgroup_ops
python setup.py develop
```
If any header files could not be found, run the following commands. 
```
python setup.py build_ext --include-dirs=$INCLUDE_PATH$
python setup.py develop
```
`$INCLUDE_PATH$` is the path to the folder containing the header files that could not be found.


## Data Preparation

(1) Download the [ScanNet](http://www.scan-net.org/) v2 dataset.

(2) Put the data in the corresponding folders. 
* Copy the files `[scene_id]_vh_clean_2.ply`,  `[scene_id]_vh_clean_2.labels.ply`,  `[scene_id]_vh_clean_2.0.010000.segs.json`  and `[scene_id].aggregation.json`  into the `dataset/scannetv2/train` and `dataset/scannetv2/val` folders according to the ScanNet v2 train/val [split](https://github.com/ScanNet/ScanNet/tree/master/Tasks/Benchmark). 

* Copy the files `[scene_id]_vh_clean_2.ply` into the `dataset/scannetv2/test` folder according to the ScanNet v2 test [split](https://github.com/ScanNet/ScanNet/tree/master/Tasks/Benchmark). 

* Put the file `scannetv2-labels.combined.tsv` in the `dataset/scannetv2` folder.

The dataset files are organized as follows.
```
PointGroup
├── dataset
│   ├── scannetv2
│   │   ├── train
│   │   │   ├── [scene_id]_vh_clean_2.ply & [scene_id]_vh_clean_2.labels.ply & [scene_id]_vh_clean_2.0.010000.segs.json & [scene_id].aggregation.json
│   │   ├── val
│   │   │   ├── [scene_id]_vh_clean_2.ply & [scene_id]_vh_clean_2.labels.ply & [scene_id]_vh_clean_2.0.010000.segs.json & [scene_id].aggregation.json
│   │   ├── test
│   │   │   ├── [scene_id]_vh_clean_2.ply 
│   │   ├── scannetv2-labels.combined.tsv
```

(3) Generate input files `[scene_id]_inst_nostuff.pth` for instance segmentation.
```
cd dataset/scannetv2
python prepare_data_inst.py --data_split train
python prepare_data_inst.py --data_split val
python prepare_data_inst.py --data_split test
```

## Training
```
CUDA_VISIBLE_DEVICES=0 python train.py --config config/pointgroup_run1_scannet.yaml 
```
You can start a tensorboard session by
```
tensorboard --logdir=./exp --port=6666
```

## Inference and Evaluation

(1) If you want to evaluate on validation set, prepare the `.txt` instance ground-truth files as the following.
```
cd dataset/scannetv2
python prepare_data_inst_gttxt.py
```
Make sure that you have prepared the `[scene_id]_inst_nostuff.pth` files before. 

(2) Test and evaluate. 

a. To evaluate on validation set, set `split` and `eval` in the config file as `val` and `True`. Then run 
```
CUDA_VISIBLE_DEVICES=0 python test.py --config config/pointgroup_run1_scannet.yaml
```
An alternative evaluation method is to set `save_instance` as `True`, and evaluate with the ScanNet official [evaluation script](https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts/3d_evaluation/evaluate_semantic_instance.py).

b. To run on test set, set (`split`, `eval`, `save_instance`) as (`test`, `False`, `True`). Then run
```
CUDA_VISIBLE_DEVICES=0 python test.py --config config/pointgroup_run1_scannet.yaml
```

c. To test with a pretrained model, run
```
CUDA_VISIBLE_DEVICES=0 python test.py --config config/pointgroup_default_scannet.yaml --pretrain $PATH_TO_PRETRAIN_MODEL$
```
