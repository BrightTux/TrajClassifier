"""
Process an image that we can pass to our networks.
"""
from keras.preprocessing.image import img_to_array, load_img
import numpy as np

def process_image(image, target_shape, add_noise):
    """Given an image, process it and return the array."""
    # Load the image.
    h, w, _ = target_shape

    bool_addnoise = add_noise

    mask_img = './img_mask.jpg'

    if (_ == 3):
        image = load_img(image, target_size=(h, w))
        mask = load_img(mask_img, target_size=(h, w))
    elif (_ == 1):
        image = load_img(image, grayscale=True, target_size=(h, w))
        mask = load_img(mask_img, grayscale=True, target_size=(h, w))

    else:
        print("Warning ... unsupported number of channels")


    # Turn it into numpy, normalize and return.
    img_arr = img_to_array(image)
    mask_arr = img_to_array(mask)


    x = (img_arr / 255.).astype(np.float32)
    x_mask = (mask_arr / 255.).astype(np.float32)

    x = x*x_mask
    #print(x.shape)


    if(bool_addnoise):
        noise_factor = 0.01
        x = x + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x.shape)
        x = x*x_mask
        x = np.clip(x, 0., 1.)



    return x
