
import albumentations as A
import skimage

def album(img_arr, num_aug_per_img = 5, img_path, lbl_path):
    """
    INPUTS  : Image array, number of augmentations per image, path to save imags, path to save labels
    OUTPUTS : Saves generated images and labels in provided paths. returns 1 on completion
    MODULES : albumentations as A, skimage
    """
    assert len(img_arr.shape) == 4, "Input array must be an array of images (Hint: Shape should be 4-d)"

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

    for i in range(len(img_arr)):
        for j in range(num_aug_per_img):
            # Augment an image
            transformed = transform(image=img_rgb, mask=lbl_rgb)
            transformed_image = transformed["image"]
            transformed_label = transformed["mask"]
            # Name the images
            image_name = str(i) + "_" + str(j) + ".png"
            label_name = str(i) + "_" + str(j) + ".png"
            # Save the images
            skimage.io.imsave(img_path + image_name, transformed_image)
            skimage.io.imsave(img_path + label_name, transformed_label)

    return 1