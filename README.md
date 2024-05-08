# TutteNet 
Code for CVPR 2024 paper TutteNet: Injective 3D Deformations by Composition of 2D Mesh Deformations. 

[Bo Sun](https://sites.google.com/view/bosun/home), [Thibault Groueix](https://imagine.enpc.fr/~groueixt/), [Chen Song](https://www.cs.utexas.edu/~song/), [Qixing Huang](https://www.cs.utexas.edu/~huangqx/), and [Noam Aigerman](https://noamaig.github.io/)

[[Arxiv]()] [[Project Page]()]

## Introduction 
![overview](images/teaser.pdf)

This work proposes a novel representation of injective deformations of 3D space, which overcomes existing limitations of injective methods, namely inaccuracy, lack of robustness, and incompatibility with general learning and optimization frameworks. Our core idea is to reduce the problem to a “deep” composition of multiple 2D mesh-based piecewise-linear maps. Namely, we build differentiable layers that produce mesh deformations through Tutte’s embedding (guaranteed to be injective in 2D), and compose these layers over different planes to create complex 3D injective deformations of the 3D volume. We show our method provides the ability to efficiently and accurately optimize and learn complex deformations, outperforming other injective approaches. As a main application, we produce complex and artifact-free NeRF and SDF deformations.

## Code Overview 
This codebase contains two parts: (1). NeRF deformation (`./nerf_deformation` folder) and (2). SMPL fitting and learning (`./fitting_learning` folder). 
NeRF deformation part contains code for NeRF training, deformation, and rendering with Instant-NGP and Nerfacto methods. Fitting and learning parts contains code for SMPL human body deformation. 
Detailed instructions can be found in their own folders. 
For the installation instructions, please follow the installation part in `./nerf_deformation` folder. 
