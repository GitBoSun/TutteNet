## About this Branch 
This branch includes code for SMPL fitting and learning experiments, which has a clearer view of our TutteNet. 

## Installation 
This code is developed based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet), which is a moduled codebase. To install it, run 
```
python setup.py develop
```

### Install and Download SMPL Body Models
Download the SMPL model version 1.1.0 from their [official website](https://smpl.is.tue.mpg.de/index.html) and update the SMPL path in `pcdet/datasets/smpl/image_smpl_dataset.py` L31. 

### Install torch_sparse_solve 
Please install `torch_sparse_solve` following the official [Github Repo](https://github.com/flaport/torch_sparse_solve). 

### Install PyMesh 
Please install `pymesh` following their [website](https://pymesh.readthedocs.io/en/latest/installation.html). 

### Install Other TutteNet Dependancies 
Run
```
pip install -r requirements.txt 
```

## Fitting Experiment

Open `fitting` folder and run 
```
python optimize_one_pair.py
```
This python script contains everything for our TutteNet and it should provide you a clear view of our method. 

## Learning Experiment 

### Dataset Download 
You can download the dataset [here](https://www.dropbox.com/scl/fi/kyl3mupocnbx4z6tm5c0a/learning_data.zip?rlkey=7ahqvneo3yygfkxbg4nh1pn6s&st=38c93rvf&dl=0). 

### Training
```
./multi_gpu_scripts/train.sh
```

### Testing 
```
./multi_gpu_scripts/test.sh
```

### Pre-trained Model Checkpoint 
You can download the pre-trained learning checkpoint [here](https://www.dropbox.com/scl/fi/cq3ec2m83dzewjk66z4xr/checkpoint_epoch_2000.pth?rlkey=oaxeop1o8sq9bky4rzsmu510e&st=6eejpmv3&dl=0). 

