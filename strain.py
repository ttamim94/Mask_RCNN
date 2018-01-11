import os
import sys
import cv2
import json
import coco
import math
import utils
import random
import visualize
import skimage.io
import matplotlib
import numpy as np
import model as modellib
import matplotlib.pyplot as plt

from PIL import Image
from config import Config

#%matplotlib inline 

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logo_models")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "logo_imgs")

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()

class LogosConfig(Config):
    """Configuration for training on the logos dataset.
    Derives from the base Config class and overrides values specific
    to the logos dataset.
    """
    # Give the configuration a recognizable name
    NAME = "logos"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5
    
#config.display()

#print(config.IMAGE_SHAPE[0])
#print(config.IMAGE_SHAPE[1])

class LogosDataset(utils.Dataset):

    def import_images(self, logo):

        form_images = np.array(
            [cv2.imread("./test_imgs/test_form.jpg")],
            dtype=np.uint8
        )

    def load_logos(self, count, height, width):
        
        self.add_class("logos", 1, "b2bank")
        self.add_class("logos", 2, "cibc")
        self.add_class("logos", 3, "bmo")

        for i in range(count):
            bg_color, shapes = self.randomize_image(height, width)

    def augment_image(self, height, width, logo='cibc'):

        min_buffer = 20

        s = random.randomint(buffer, height//4)         
        x = random.randomint(buffer, width - buffer -1)
        y = random.randomint(buffer, height - buffer -1)
        
        return color, (x, y, s)

    def randomize_logos(self, width, height, logo='cibc', ext='.jpg'):

        img = Image.open('./logo_imgs/mins/'+logo+ext)
        img_w, img_h = img.size

        bg_color = tuple(np.array([random.randint(0, 255) for _ in range(3)]))
        background = Image.new('RGB', (1440, 900), bg_color)
        bg_w, bg_h = background.size
        
        print( random.randint(1, bg_w - img_w) )

        offset = (
            (bg_w - img_w) // random.randint(1, min(4, bg_w - img_w)), 
            (bg_h - img_h) // random.randint(1, min(4, bg_h - img_h))
        )
        print("working")

        background.paste(img, offset)
        background.save('out.png')
        
if __name__ == "__main__":

    training_dataset = LogosDataset()
    training_dataset.randomize_logos(256, 256, 'cibc')


