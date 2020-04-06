# AA+Sim 3A

## TrainGan

### Run 6A on GPU

Run original cryoGAN traininng sequence:
ssh rack-wolfson-g02
cd /specific/netapp5_2/iscb/wolfson/Mark/git/work_from_home/NNproject/code/python/
setenv LD_LIBRARY_PATH /usr/local/lib/cuda-10.0.130/lib64/:/usr/local/lib/cudnn-9.0-v7/lib64/
source /specific/netapp5_2/iscb/wolfson/Mark/v_env/p36_tf/bin/activate.csh

### Change VX to 1 and Resolution to 3

Copied code to the new location
in dataset_loader.py changed VX_SIZE = 1.0, RESOLUTION = 3



### Create DB for 3A (from AAnchor)

### Run 3A on GPU

### Test - See the boxes

## Create Dataset

### Create DB (pdb) from Rotamers

### Run GAN - create Maps

## Train AAnchor

### Train Old Version (Old DB)

### Train on New DB

## Calibrate AAnchor

## Test AAnchor

### Run on new Molecules

*XMind: ZEN - Trial Version*

*XMind: ZEN - Trial Version*



*XMind: ZEN - Trial Version*