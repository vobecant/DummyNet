# DummyNet
Official implementation of paper "Artificial Dummies for Urban Dataset Augmentation". [[arXiv paper]](https://arxiv.org/abs/2012.08274)

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
mkdir data
tar -zxvf data.tar.gz
```
Then, extract the weights in `./data/weights` folder.
```
cd data/weights
tar -zxvf DummyNetAAAI_files.tar.gz
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
