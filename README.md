# Carrier pocket effect at charged domain walls in wurtzite ferroelectrics -- AI-assisted epitaxial growth
## RHEED Analyzer based on Neurosim

This repository contains the work of RHEED image spotty level scorer based on Neurosim framework [https://github.com/neurosim/DNN_NeuroSim_V2.1.git]. The author is Dalei Jiang, from Prof. Zetian Mi's group of University of Michigan, Ann Arbor.
## Dataset packing guildline
The dataset packing algorithms is stored in the folder *DatasetOperator*, dataset path is set to be ../resource, the same-level folder of workspace.

### From **video** to **Proper Length Clips**
File *Splitter.py* split the Long video file stored in folder ./video into 15-second clips. Since video operation usually costs quite big computation source, this step optimizes further processes. To use the function, firstly place the target video into folder *video* in the same level as the current repository. After changing the clip time length, the clips will be stored in newly generated folder inside *video*. In our project, the video is named as **Material name_Score_Unique Index**
tips: If one video contains multiple growth stage, and are given different scores, please use function *ScoreStageDivider.py* to cut one video to several ones, and give each one its proper score by renaming.

### Crystal Orientation and Central Spot Location functions
In this study, we focus on extracting RHEED (Reflection High-Energy Electron Diffraction) images corresponding to the crystallographic directions $\left[1\bar{1}20\right]$ and $\left[10\bar{1}0\right]$. These are two high-symmetry in-plane directions in hexagonal crystal systems, such as GaN. RHEED patterns observed along these directions are particularly informative, as they provide clear diffraction features that are highly sensitive to surface morphology, lattice constants, and reconstruction phases. The $\left[1\bar{1}20\right]$ direction aligns with the a-axis, while $\left[10\bar{1}0\right]$ corresponds to the m-axis of the hexagonal unit cell.
To facilitate structure analysis and reproducibility, we extract RHEED frames from video sequences specifically captured along these two orientations. These part of functions are stored in *ParameterLab.ipynb*. The **Brightness Distribution Analyzer** can determine the location of central spot in the processed image automatically, and **Crystallographic Directions Location**  is used to select the first index position of $\left[1\bar{1}20\right]$ and $\left[10\bar{1}0\right]$. Since the substrate is rotating with constant angular speed of 0.4 rps, the following images can be extracted by applying certain step on video. The result packing file will be stored in NPY array file.

### From **Proper Length Clips** to **NPY Files**
The file *DataGenerator* in the folder **DatasetOperator** is working to transform the video file into a NPY file. There are seven parameters need to input.
1. Material type: Determines the material folder to reach. 
2. Score: The target spotty level video. When we are evaluating new video without actual score, the target will be marked as "TEST", and the Score is set as default "0".
3. Index: The videos with same material type and the same score are stored in a same folder, input actual index number to get related video. 
4. Video Number: In the previous step, we have divide a long video into clips to optimize following function complexity, mainly showing in this step. Here, input the number of videos cut from the previous step. The function will go through all the clips and truncate the images and pack as a whole array. 
The user can also use VideoLength // ClipLength + 1 as Video Number.
5. offerset1: The first index location of $\left[10\bar{1}0\right]$, the following images will be fetched by constant step $fraction{FrameRate, RotationSpeed}$

### From resized dataset to convoluted result (Network Input)
The resized dataset is set to be a 4-dimentional array. Using *DataConv.py* to transfer resized dataset to convoluted feature map. The result feature map will be input into the network. After trying the combination of different kernels, the best match will be the vertical Sobel and horizontal Sobel. The detail of different kernel combinations are shown in Visualization/Binprinter.ipynb

## Model Training
Make sure the target file is stored in the same-level folder **resource**. In file *Train.py*, set the dataset label and all the parameters properly to fetch the dataset with correct preprocessing kernels. The result model will be store in folder **model**, which is also set in the same level folder with current repository.
