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
from scipy.ndimage import filters

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

    def convert_image_to(self, img, new_format='opencv'):

        if new_format == 'opencv':
            new_img =  cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            #cv2.imshow('Demo Image', formatted_img)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            
            return new_img
        elif new_format == 'pil':
            new_img = Image.fromarray(img)
            return new_img

    def load_logos(self):
        
        self.add_class("logos", 1, "b2bank")
        self.add_class("logos", 2, "cibc")
        self.add_class("logos", 3, "bmo")

        print(self.image_ids)
        print(self.image_info)
        print(self.class_info)

        for idx, logo_class in enumerate(self.class_info):
            print(logo_class['name'])

        #for i in range(count):
        #    logo = self.randomize_logo(width, height)
        #    self.add_image("logos", image_id=i, path=None, 
        #                    width=width, height=height,
        #                    logos=logos)

    def augment_image(self, height, width, logo='cibc'):

        min_buffer = 20

        s = random.randomint(buffer, height//4)         
        x = random.randomint(buffer, width - buffer -1)
        y = random.randomint(buffer, height - buffer -1)
        
        return color, (x, y, s)

    def randomize_logo(self, width, height, logo='cibc', ext='.jpg'):

        img = Image.open('./logo_imgs/mins/'+logo+ext)
        img_w, img_h = img.size
        print(img_w, img_h)

        bg_size = (1440, 900)
        bg_color = tuple(np.array([random.randint(0, 255) for _ in range(3)]))
        background = Image.new('RGB', bg_size, bg_color)
        bg_w, bg_h = background.size
        background.putdata(self.white_noise(bg_w, bg_h))        

        offset = (
            (bg_w - img_w) // random.randint(min(2, bg_w - img_w), min(6, bg_w - img_w)), 
            (bg_h - img_h) // random.randint(min(2, bg_w - img_w), min(6, bg_h - img_h))
        )
        print("working")

        opencvimg = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        

        #img = Image.fromarray(self.gaussian_noise(opencvimage))

        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.imshow('Demo Image', opencvimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        
        #cv2.line(opencvimg,(0,0),(200,300),(255,255,255),50)        
        #cv2.rectangle(img,(500,250),(1000,500),(0,0,255),15)
        #cv2.circle(img,(447,63), 63, (0,255,0), -1)
        #pts = np.array([[100,50],[200,300],[700,200],[500,100]], np.int32)
        #pts = pts.reshape((-1,1,2))
        #cv2.polylines(img, [pts], True, (0,255,255), 3)
        #font = cv2.FONT_HERSHEY_SIMPLEX
        #cv2.putText(img,'OpenCV Tuts!',(10,500), font, 6, (200,255,155), 13, cv2.LINE_AA)
        #cv2.imshow('image',img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()



        #m = (50, 50, 50)
        #s = (50, 50, 50)
        #img = cv2.randn(img, m, s)

        background.paste(img, offset)

        background.save('./train/cibc.png')

    def randomize_forms(self, logo='cibc', ext='.jpg'):

        img = Image.open('./logo_imgs/'+logo+ext)
        img_w, img_h = img.size
        img.putdata(self.white_noise(img_w, img_h))    
        img.save('out.png')    

        #offset = (
        #    (bg_w - img_w) // random.randint(min(2, bg_w - img_w), min(6, bg_w - img_w)), 
        #    (bg_h - img_h) // random.randint(min(2, bg_w - img_w), min(6, bg_h - img_h))
        #)
        #print("working")

    def gaussian_noise(self, img, radius=5):
        img2 = np.zeros(img.shape)
        for i in range(3):
            img2[:,:,i] = filters.gaussian_filter(img[:,:,i], 1)
        img2 = np.array(img2, 'uint8')
        return img2

    def white_noise(self, width, height):
        random_grid = map(lambda x: (
                int(random.random() * 256),
                int(random.random() * 256),
                int(random.random() * 256)
            ), [0] * width * height)
        return list(random_grid)

if __name__ == "__main__":

    training_dataset = LogosDataset()
    #training_dataset.randomize_logos(1024, 1024, 'bmo')
    training_dataset.load_logos()

    #print(config.IMAGE_SHAPE[0])
    #print(config.IMAGE_SHAPE[1])

    #training_dataset.load_shapes(1, config)

    #training_dataset.randomize_forms()
    #training_dataset.white_noise(1440, 900)

