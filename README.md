# Carrier pocket effect at charged domain walls in wurtzite ferroelectrics -- AI-assisted epitaxial growth
## RHEED Analyzer based on Neurosim

This repository contains the work of RHEED image spotty level scorer based on Neurosim framework [https://github.com/neurosim/DNN_NeuroSim_V2.1.git]. The author is Dalei Jiang, from Prof. Zetian Mi's group of University of Michigan, Ann Arbor.
## Dataset packing guildline
The dataset packing algorithms is stored in the folder *DatasetOperator*

### From resized dataset to convoluted result (Network input)
The resized dataset is set to be a 4-dimentional array. Using *DataConv.py* to transfer resized dataset to convoluted feature map. The result feature map will be input into the network. After trying the combination of different kernels, the best match will be the vertical Sobel and horizontal Sobel. The detail of different kernel combinations are shown in Visualization/Binprinter.ipynb
