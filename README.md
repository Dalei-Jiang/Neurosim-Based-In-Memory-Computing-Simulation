# Carrier pocket effect at charged domain walls in wurtzite ferroelectrics -- AI-assisted epitaxial growth
## RHEED Analyzer based on Neurosim

This repository contains the work of RHEED image spotty level scorer based on Neurosim framework [https://github.com/neurosim/DNN_NeuroSim_V2.1.git]. The author is Dalei Jiang, from Prof. Zetian Mi's group of University of Michigan, Ann Arbor.
## Dataset packing guildline
The dataset packing algorithms is stored in the folder *DatasetOperator*, dataset path is set to be ../resource, the same-level folder of workspace.

### From video to NPY file
File *Splitter.py* split the Long video file stored in folder ./video into 15-second clips. Since video operation usually costs quite big computation source, this step optimizes further processes. To use the function, firstly place the target video into folder *video* in the same level as the current repository. After changing the clip time length, the clips will be stored in newly generated folder inside *video*.

### From resized dataset to convoluted result (Network input)
The resized dataset is set to be a 4-dimentional array. Using *DataConv.py* to transfer resized dataset to convoluted feature map. The result feature map will be input into the network. After trying the combination of different kernels, the best match will be the vertical Sobel and horizontal Sobel. The detail of different kernel combinations are shown in Visualization/Binprinter.ipynb
