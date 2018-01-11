import cv2
import numpy as np
import imgaug as ia

from imgaug import augmenters as iaa
from skimage.transform import resize

def save_image(name, image):
    cv2.imwrite(name,image)
    print("saved image")


def resize_image_with_padding(img, new_shape, fill_cval=None, order=1):
    if fill_cval is None:
        fill_cval = np.max(img)
    ratio = np.min([n / i for n, i in zip(new_shape, img.shape)])
    print(ratio)
    interm_shape = np.rint([s * ratio for s in img.shape]).astype(np.int)
    print(interm_shape)

    interm_img = resize(img, interm_shape, order=order, cval=fill_cval)

    new_img = np.empty(new_shape, dtype=interm_img.dtype)
    new_img.fill(fill_cval)

    pad = [(n - s) >> 1 for n, s in zip(new_shape, interm_shape)]
    print(new_shape)
    print(interm_shape)
    print(pad)
    new_img[[slice(p, -p, None) if 0 != p else slice(None, None, None) 
             for p in pad]] = interm_img

    return new_img

ia.seed(100)

images = np.array(
    [cv2.imread("./images/bmo_logo_empty_form.jpg")],
    dtype=np.uint8
)

cnt = 0
for idx, image in enumerate(images):
    print(image.shape)
    resized_image = resize_image_with_padding(img=image, new_shape=(400, 400, 2), fill_cval=np.max(image)*0.95)
    save_image("resized_img"+str(cnt)+".jpg", image)
    cnt += 1
# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second
# image.
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# Define our sequence of augmentation steps that will be applied to every image.
seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips
    iaa.Crop(percent=(0, 0.1)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    # Strengthen or weaken the contrast in each image.
    iaa.ContrastNormalization((0.75, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8)
    )
], random_order=True) # apply augmenters in random order

augmented_images = seq.augment_images(images)

for idx, augmented_image in enumerate(augmented_images):
    save_image("aug_img"+str(idx)+".jpg", augmented_image)
