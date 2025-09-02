import numpy as np
import os
from tqdm import tqdm
import cv2
from skimage.io import imread, imsave
import albumentations as A
import matplotlib.pyplot as plt

def shuffle_data(X, Y, debug_sw = 0):
    """ 
    INPUT : takes image arrays.
    OUTPUT: returns shuffled image arrays.
    """
    assert X.shape[0] == Y.shape[0], "Input arrays are not of same length"
    # variables to store shuffled data
    X_shuffle = np.zeros(X.shape)
    Y_shuffle = np.zeros(Y.shape)
    # generate random indices
    arr = np.arange(len(X))
    np.random.shuffle(arr)
    # create shuffled array
    for i in range(len(arr)):   
        X_shuffle[i] = X[arr[i]]
        Y_shuffle[i] = Y[arr[i]]
    return X_shuffle, Y_shuffle
    
def rotate_image(img_arr):
    """
    INPUT : Image array.
    OUTPUT: retunrns a 90 degrees clock-wise rotated image.
    """
    assert len(img_arr.shape) == 4, "Input array must be an array of images"
    rot_img = np.zeros(img_arr.shape)
    for i in range(len(img_arr)):
        rot_img[i] = cv2.rotate(img_arr[i], cv2.ROTATE_90_COUNTERCLOCKWISE).reshape(img_arr[i].shape[0], img_arr[i].shape[1], img_arr[i].shape[2])
    return rot_img
    
def plotImages(images_arr, num_plots = 5):
    fig, axes = plt.subplots(1, num_plots, figsize=(30, 30))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
def load_data(TRAIN_PATH, TEST_PATH, train_ids, test_ids, IMG_WIDTH=256, IMG_HEIGHT=256):
    """
    INPUT : Takes path of train and test folders. These folders MUST contain image, label subfolders. 
    OUTPUT: returns four arrays of train and test images.
    """
    x_train = []
    y_train = []
    x_test  = []
    y_test  = []
    
    # print('Resizing train images and masks')
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        train_img = imread(TRAIN_PATH + '/image/' + id_)
        train_img = cv2.resize(train_img, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_NEAREST)
        x_train.append(train_img)
        train_mask = imread(TRAIN_PATH + '/label/' + id_ )
        train_mask = cv2.resize(train_mask, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_NEAREST)
        y_train.append(train_mask)
        
    # print('Resizing test images and masks')
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
            test_img = imread(TEST_PATH + '/image/' + id_ )
            test_img = cv2.resize(test_img, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_NEAREST)
            x_test.append(test_img)
            test_mask = imread(TEST_PATH  + '/label/' + id_ )
            test_mask = cv2.resize(test_mask, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_NEAREST)
            y_test.append(test_mask)
    
    return np.array(x_train)/255, np.array(y_train)/255 , np.array(x_test)/255 , np.array(y_test)/255 


def album(num_aug_per_img, img_name, img, lbl, img_path, lbl_path):
    """
    INPUTS  : num_aug_per_img, img_name, img, lbl, img_path, lbl_path
    OUTPUTS : Saves generated images and labels in provided paths. returns 1 on completion
    MODULES : albumentations as A, skimage
    """
    transform = A.Compose([
        A.RandomCrop(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=0.5),
        # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5),
        # A.Cutout (num_holes=8, max_h_size=8, max_w_size=8, fill_value=0, always_apply=False, p=0.5),
        A.Emboss (alpha=(0.2, 0.5), strength=(0.2, 0.7), always_apply=False, p=0.5),
        # A.Equalize (mode='cv', by_channels=True, mask=None, mask_params=(), always_apply=False, p=0.5),
        A.FancyPCA (alpha=0.1, always_apply=False, p=0.5),
        A.GlassBlur (sigma=0.7, max_delta=4, iterations=2, always_apply=False, mode='fast', p=0.5),
        A.GaussNoise (var_limit=(10.0, 50.0), mean=0, per_channel=True, always_apply=False, p=0.5),
        A.GaussianBlur (blur_limit=(3, 7), sigma_limit=0, always_apply=False, p=0.5)
        ])
    
    for j in range(num_aug_per_img):
        # Augment an image
        transformed = transform(image=img, mask=lbl)
        transformed_image = transformed["image"]
        transformed_label = transformed["mask"]
        # Name the images
        image_name = str(img_name).replace(".", "_"+str(j)+".")
        label_name = str(img_name).replace(".", "_"+str(j)+".")
        # Save the images
        imsave(img_path + image_name, transformed_image)
        imsave(lbl_path + label_name, transformed_label)
#             cv2.imwrite(img_path + image_name, transformed_image)
#             cv2.imwrite(lbl_path + label_name, transformed_label)
    return 1
