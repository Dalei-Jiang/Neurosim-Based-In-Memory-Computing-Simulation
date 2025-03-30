# Carrier pocket effect at charged domain walls in wurtzite ferroelectrics -- AI-assisted epitaxial growth
## RHEED Analyzer based on Neurosim

This repository contains the work of RHEED image spotty level scorer based on Neurosim framework [https://github.com/neurosim/DNN_NeuroSim_V2.1.git]. The author is Dalei Jiang, from Prof. Zetian Mi's group of University of Michigan, Ann Arbor.

## File lists

1. Train.py
This file is the main body for model training. Dataset and network settings are in the folder cifar, which is set in the parent program [https://github.com/neurosim/DNN_NeuroSim_V2.1.git]. To replace the network structure or dataset, please move the file to 'resource' folder, and change the dataset name in './cifar/dataset' or './cifar/Network'.

2. util.py
Containing all necessary small functions needed by other files. The email notification module is here. To activate it, please fulfill all "TODO" with own email settings.

3. Img_generator.py
This part can generate all the visualization images needed, such as training/validation loss and accuracy curves, confusion matrices, model parameter distributions and so on.
