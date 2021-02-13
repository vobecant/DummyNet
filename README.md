# DummyNet
Official implementation of paper "Artificial Dummies for Urban Dataset Augmentation". [[arXiv paper]](https://arxiv.org/abs/2012.08274)

Videos can be found [here](https://virtual.2021.aaai.org/paper_AAAI-9933.html).

## Setup
**Note**: The code is tested only on Linux distributions.

Run
```
git clone https://github.com/vobecant/DummyNet.git
cd DummyNet
conda env create -f environment.yml
conda activate DummyNet
```
to create and activate the new conda environment.

## Data
First, please download sample data and extract it to `./data`. 
```
wget https://data.ciirc.cvut.cz/public/projects/DummyNet/data.tar.gz
tar -zxvf data.tar.gz
```

The structure of the ./data folder should be:
```
data/
  YBB/
    gan_test.json
    ...
    test_samples_100.th
  weights/
    GAN_GEN_4.pth
    ...
    MASK_ESTIMATOR.pth
```

## Run example

### NightOwls
To augment the NightOwls dataset, run:
```
python augment_nightowls.py ./data/weights ${SAVE_DIR} ./data/YBB/nightowls_bbs
```
The script takes three arguments. You need to set
- `SAVE_DIR`: directory where the extended dataset will be saved

### CityPersons
To augment the CityPersons datasets, run:
```
python augment_cs.py ./data/weights/ ${CITYSCAPES_DIR} ${SAVE_DIR}
```
The script takes three arguments:
- `weights_dir`: path to the directory with weights
- `CITYSCAES_DIR`: path to the directory with Cityscapes dataset and CityPersons dataset
- `SAVE_DIR`: directory where the extended dataset will be saved



## Using Pose Generator
To use the Pose Generator, please refer to `README_pose_generator.txt`.

Required packages:
- numpy 1.16.5
- matplotlib 3.1.1
- jsonschema 3.0.2
- sklearn 0.21.2 (0.21.3 generates warning, but works too)
- joblib 0.13.2
- dill 0.3.3

First, you need to download [joints_pca_etc.npz](https://data.ciirc.cvut.cz/public/projects/DummyNet/joints_pca_etc.npz) and [pca_per_cluster.zip](https://data.ciirc.cvut.cz/public/projects/DummyNet/pca_per_cluster.zip). To do this, you can run
```
wget https://data.ciirc.cvut.cz/public/projects/DummyNet/joints_pca_etc.npz
wget https://data.ciirc.cvut.cz/public/projects/DummyNet/pca_per_cluster.zip
```
and unzip it using
```
unzip pca_per_cluster.zip
```
Then set the paths in `pose_generator.py` and run.


## Pretrained detector weights.
You can download CSP detector weights trained on CityPersons dataset [here](https://data.ciirc.cvut.cz/public/projects/DummyNet/csp_best.hdf5)
