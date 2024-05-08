## About this Branch
This branch includes code for NeRF training, elastic deformation and rendering in the Nerfstudio codebase. 
We provide deformation for two methods: Instant-NGP and Nerfacto, where the former is more suitable for clean objects and the latter is better for wide boundless NeRFs (e.g. phone-captured NeRFs). 
[![Examples of Elastic NeRF Deformation](https://img.youtube.com/vi/LJsfEFTT_BY/default.jpg)](https://youtu.be/LJsfEFTT_BY)

## Installation 
### Install Nerfstudio 
Please follow the installation instructions on the official website [here](https://docs.nerf.studio/quickstart/installation.html). 

Then replace the conda nerfstudio package with our customized package: 
```
mv path_to_conda/envs/python3/lib/python3.8/site-packages/nerfstudio path_to_conda/envs/python3/lib/python3.8/site-packages/nerfstudio_or
ln -s nerfstudio path_to_conda/envs/python3/lib/python3.8/site-packages/nerfstudio
```

### Install torch_sparse_solve 
Please install `torch_sparse_solve` following the official [Github Repo](https://github.com/flaport/torch_sparse_solve). 

### Install Other TutteNet Dependancies 
Run
```
pip install -r requirements.txt 
```

## Data Download
### NeRF Dataset 
You can download the NeRF datasets we used in our paper [here](https://www.dropbox.com/scl/fi/xlqv86i5b3lbp0wq3oxzq/TutteNet_NeRF_data.zip?rlkey=wr4ydu1kyjuw92g6ud80yr4b0&st=vtupn054&dl=0). 
They contains the official nerf_synthetic dataset, data from SPIDR, and our own data. 

Please put this `data` folder under the root path. 

### Deformation Handles
We also provide some example deformation handles that you can use in your NeRF deformation. You can download them [here](https://www.dropbox.com/scl/fi/zgsx9odde7js5ok3cmgbc/nerf_deformation.zip?rlkey=psxadbgwuavorzr78m93ypxcl&st=fqr7wk2c&dl=0). 

Please put these folders under the root path. 

## Pipeline 
The whole pipeline consists of 3 steps: (1). NeRF pre-training, (2), NeRF deformation, and (3). Rendering deformed NeRFs. Below we will use lego instance in the nerf_synthetic dataset and Instant-NGP model as an example. You can download the training checkpoints [here](https://www.dropbox.com/scl/fi/f549lose4hzkzte6z218o/nerf_checkpoints.zip?rlkey=72ijbgkmwiifw4686jmepo1ev&st=tc4dw050&dl=0) and directly run the third step. 

### Step 1: NeRF Training
Run
```
./nerf_train.sh
```

### Step 2: Elastic Deformation with Given Constriants 
```
python apply_deformation.py
```

Note that this script used the prepared deformation handles you downloaed [here] and constraints at the head of this file. You can create your own handles and handle constriants with Meshlab or any point cloud selection tools. 

### Step 3: Render the Deformed NeRF

For instant-NGP, run
```
./nerf_train_with_deformation.sh
```
This step is to update the density field of the Instant-NGP model so that it can sample points around the deformed shape when rendering images. 

Then render the deformed images: 
```
./nerf_export_image_deform.sh
```
If you want to render the original images for comparison, run: 
```
./nerf_export_image.sh
```

For Nerfacto, directly run the export script to get the deformed images. (You have to train your Nerfacto model and deform its density points first)
```
./nerf_export_image_deform.sh
```

