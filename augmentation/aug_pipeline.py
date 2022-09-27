import os.path
import random
from time import sleep

from tqdm import tqdm

import albumentations as A
import cv2
from matplotlib import pyplot as plt

from common.ClassFile import ClassFile


def visualize(_image):
    try:
        plt.figure(figsize=(10, 10))
        plt.axis('off')
        plt.imshow(_image)
        plt.pause(1)
    except Exception:
        raise IOError()


def load(path, doc=None, ext_list=[]):
    print(f"Loading data for augmentation from [{path}]", end="")
    _image_list = list()
    if doc is None:
        count = 0
        for ext in ext_list:
            image_file_list = ClassFile.list_files_like(path, f".{ext}")
            random.shuffle(image_file_list)
            for image_file in image_file_list:
                _image = cv2.imread(image_file)
                if _image is not None:
                    _image = cv2.cvtColor(_image, cv2.COLOR_BGR2RGB)
                    _image_list.append((image_file, _image))
                    count += 1
                    if count > 100:
                        print(".", end="")
                        count = 0
                else:
                    print("X", end="")
    else:
        _image = cv2.imread(os.path.join(path, doc))
        if _image is not None:
            _image = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)
            _image_list.append((doc, _image))
            print(".", end="")
        else:
            print("X", end="")
    print()
    sleep(3)

    return _image_list


def transform(file, image, count):
    _transform = A.Compose([
        A.RandomFog(p=0.05),
        # A.RandomShadow(p=0.4),
        A.Sharpen(p=0.3),
        A.VerticalFlip(p=0.01),
        A.HorizontalFlip(p=0.01),
        A.Rotate(limit=(-10, 10), p=0.8),
        A.RandomToneCurve(p=0.2),
        A.GaussNoise(p=0.4),
        # A.GlassBlur(p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
    ])
    augmented_image = list()
    augmented_name = list()
    for _ in range(1, count+1):
        augmented_name.append(file)
        augmented_image.append(_transform(image=image)['image'])
    return augmented_name, augmented_image


def main():
    random.seed(42)
    AUG_FACTOR = 10
    IMG_EXT = ["jpg", "png", "jpeg"]
    image_list = load(f"{image_path}", ext_list=IMG_EXT)
    print("Augmenting dataset...")
    loop = tqdm(image_list)
    for file, image in loop:
        # original image
        f_path, f_name = os.path.split(file)
        image_train = os.path.join(f_path, "aug")
        os.makedirs(image_train, exist_ok=True)

        # augmented image
        augmented_name_list, augmented_image_list = \
            transform(file=file, image=image, count=AUG_FACTOR)
        # visualize(augmented_image_list[0])
        for idx, (augmented_name, augmented_image) in \
                enumerate(zip(augmented_name_list, augmented_image_list)):
            aug_f_path, aug_f_name = os.path.split(augmented_name)
            aug_f_name, aug_f_ext = os.path.splitext(aug_f_name)
            aug_f_name = f"{aug_f_name}_aug{idx}{aug_f_ext}"
            cv2.imwrite(os.path.join(image_train, aug_f_name), augmented_image)


if __name__ == '__main__':
    image_path = rf"C:\Users\JoseAntonioFernandez\Desktop\IMAGE\classification\dataset\train\NEW"
    main()
