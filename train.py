from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from PIL import Image
import shutil
from skimage.filters import threshold_otsu
from skimage.morphology import dilation, disk, erosion

preceding_path = "/blue/eel5840/justin.rossiter/final_project_team_square_root"

plt.ion()   # interactive mode
#https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

def generateImageAndLabel(split, i, idx):
    image_grayscale = dataset[:, :, idx]
    image_dilated = dilation(image_grayscale) #Dilation
    image_erosion = erosion(image_dilated) #Image Erosion
    radius = 15 #Otsu thresholding
    footprint = disk(radius)
    local_otsu = rank.otsu(image_erosion, footprint)
    image_local_otsu_threshold = image_erosion >= local_otsu
    image_bounding = 1-image_local_otsu_threshold
    left_edge = np.where(np.array([np.sum(image_bounding[:, i]) for i in range(300)]) > 0)[0]
    right_edge = np.where(np.array([np.sum(image_bounding[:, i]) for i in range(300)]) > 0)[-1]
    top_edge = np.where(np.array([np.sum(image_bounding[i, :]) for i in range(300)]) > 0)[0]
    bottom_edge = np.where(np.array([np.sum(image_bounding[i, :]) for i in range(300)]) > 0)[-1]
    width = (right_edge - left_edge)/300.0
    height = (bottom_edge - top_edge)/300.0
    x_mu = (right_edge + left_edge)/600.0
    y_mu = (bottom_edge + top_edge)/600.0
    print(image_local_otsu_threshold)
    image = Image.fromarray(image_grayscale, mode = 'L').resize((320, 320))
    image.save(os.path.join(preceding_path, split, "images", str(i) + "_" + str(idx) + ".jpg"))
    with open(os.path.join(preceding_path, split, "labels", str(i) + "_" + str(idx) + ".txt"), "w") as f:
        f.write(str(i) + " " + str(x_mu) + " " + str(y_mu) + " " + str(width) + " " + str(height))

if __name__ == "__main__":

    test_size = 0.25

    dataset = np.load(os.path.join(preceding_path, 'data_train.npy')).reshape((300, 300, 9032))
    labels = np.load(os.path.join(preceding_path, 't_train_corrected.npy'))
    if os.path.exists(os.path.join(preceding_path, "train")):
        shutil.rmtree(os.path.join(preceding_path, "train"))
    if os.path.exists(os.path.join(preceding_path, "test")):
        shutil.rmtree(os.path.join(preceding_path, "test"))
    os.makedirs(os.path.join(preceding_path, "train", "images"))
    os.makedirs(os.path.join(preceding_path, "train", "labels"))
    os.makedirs(os.path.join(preceding_path, "test", "images"))
    os.makedirs(os.path.join(preceding_path, "test", "labels"))

    train_indices, test_indices = train_test_split(range(9032), test_size=test_size, shuffle=False)

    for i in range(10):
        for idx in np.where(labels == i)[0]:
            if idx in train_indices:
                generateImageAndLabel("train", i, idx)
            else:
                generateImageAndLabel("test", i, idx)
                
    for idx in np.where(labels == -1)[0]:
        if idx in train_indices:
            generateImageAndLabel("train", 10, idx)
        else:
            generateImageAndLabel("test", 10, idx)

