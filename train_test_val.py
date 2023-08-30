import splitfolders as split_folders
import os
import numpy as np
from sklearn.model_selection import train_test_split

#inputDataset='C:/Ph.d/4. Data Set/Corrected_IMFDB_(128_128)_V1/'

#inputDataset='C:/Ph.d/15. facenet/mtcnn_icdfd/'
inputDataset='D:/Programs/augmented_IIITD_V2'

# inputDataset='D:/DataSets/Disguise Face DS_V2/All_Images/'
#output='C:/Ph.d1/17. pca on iiitd/4. pca/IMFDB_40_60/'
#output='C:/Ph.d/15. facenet/mtcnn_icdfd_splitt/'
output='D:\Programs\Project_new_vgg\dataset'


split_folders.ratio(inputDataset, output=output, seed=1332, ratio=(.6, 0.2, 0.2)) # ratio of split are in order of train/val/test. You can change to whatever you want. For train/val sets only, you could do .75, .25 for example.
#split_folders.ratio(inputDataset, output=output, seed=42, ratio=(0.7, 0.15,0.15),group_prefix=None)# splits into train and val


