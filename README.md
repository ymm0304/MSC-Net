Requirements
---
 Python>=3.7
 Pytorch>-1.7
 
 Datasets
 ---
 We use these two publicly available datasets for segmentation training and coloring training.Please place them in the corresponding folder.
 
 https://github.com/jianweiguo/SpecularityNet-PSD (PSD)
 
 https://github.com/fu123456/SHIQ (SHIQ)

 To place the corresponding input image and its mask image in the corresponding code position.

 1:Highlight Feature Extraction(1-SHSNet) and Highlight Elimination
---
Training segmentation model

train.py  

Testing segmentation model

test.py


Eliminating highlights

removal.py

 2:Image Coloring and Highlight Restoration
 ---
 Training coloring model

 train.py 

 Testing coloring model

test.py

Restoring highlights

reflect.py

