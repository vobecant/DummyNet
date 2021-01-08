# DummyNet
Official implementation of paper "Artificial Dummies for Urban Dataset Augmentation"

## Setup
Run
```
conda env create -f environment.yml
conda activate DummyNet
```
to create and activate the new conda environment.

## Data
Please download sample data from https://data.ciirc.cvut.cz/public/projects/DummyNet/ and extract it to `./data`. 
Then, please extract the weights in `./data/weights` folder.

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
