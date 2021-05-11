
import functools
import os
import argparse
import time

from matplotlib import gridspec
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--content", required = True,
                    help = "Path to the content image")
    ap.add_argument("-i", "--style", required = True,
                    help = "Path to the style image")

    args = vars(ap.parse_args())
    # load the query image and show it
    content_path = args["content"]
    style_path = args["style"]

    print("TF Version: ", tf.__version__)
    print("TF-Hub version: ", hub.__version__)
    print("Eager mode enabled: ", tf.executing_eagerly())
    print("GPU available: ", tf.config.list_physical_devices('GPU'))
    print("Tensorflow built with CUDA: ", tf.test.is_built_with_cuda())

    style_transfer(content_path, style_path)

# @title Define image loading and visualization functions  { display-mode: "form" }

def crop_center(image):
    """Returns a cropped square image."""
    shape = image.shape
    new_shape = min(shape[1], shape[2])
    offset_y = max(shape[1] - shape[2], 0) // 2
    offset_x = max(shape[2] - shape[1], 0) // 2
    image = tf.image.crop_to_bounding_box(
        image, offset_y, offset_x, new_shape, new_shape)
    return image

@functools.lru_cache(maxsize=None)
def load_image_url(image_url, image_size=(256, 256), preserve_aspect_ratio=True):
    """Loads and preprocesses images."""
    # Cache image file locally.
    image_path = tf.keras.utils.get_file(os.path.basename(image_url)[-128:], image_url)
    # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
    img = plt.imread(image_path).astype(np.float32)[np.newaxis, ...]
    if img.max() > 1.0:
        img = img / 255.
    if len(img.shape) == 3:
        img = tf.stack([img, img, img], axis=-1)
    img = crop_center(img)
    img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
    return img

def load_image_path(image_path, image_size=(256, 256)):
    img = plt.imread(image_path).astype(np.float32)[np.newaxis, ...]
    if img.max() > 1.0:
        img = img / 255.
    if len(img.shape) == 3:
        img = tf.stack([img, img, img], axis=-1)
    img = crop_center(img)
    img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
    return img

def show_n(images, titles=('',)):
    n = len(images)
    image_sizes = [image.shape[1] for image in images]
    w = (image_sizes[0] * 6) // 320
    plt.figure(figsize=(w  * n, w))
    gs = gridspec.GridSpec(1, n, width_ratios=image_sizes)
    for i in range(n):
        plt.subplot(gs[i])
        plt.imshow(images[i][0], aspect='equal')
        plt.axis('off')
        plt.title(titles[i] if len(titles) > i else '')
    plt.show()

def style_transfer(content_image_path, style_image_path):

    output_image_size = 384  # @param {type:"integer"}

    # The content image size can be arbitrary.
    content_img_size = (output_image_size, output_image_size)
    # The style prediction model was trained with image size 256 and it's the
    # recommended image size for the style image (though, other sizes work as
    # well but will lead to different results).
    style_img_size = (256, 256)  # Recommended to keep it at 256.

    content_image = load_image_path(content_image_path, content_img_size)
    style_image = load_image_path(style_image_path, style_img_size)
    style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='SAME')

    show_n([content_image, style_image], ['Content image', 'Style image'])

    # Load TF-Hub module.

    hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
    hub_module = hub.load(hub_handle)

    # Stylize content image with given style image.
    # This is pretty fast within a few milliseconds on a GPU.

    outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
    stylized_image = outputs[0]

    # Visualize input images and the generated stylized image.

    show_n([content_image, style_image, stylized_image], titles=['Original content image', 'Style image', 'Stylized image'])

    file_name = 'stylized-image-slow-' + time.strftime("%m-%d-%H-%M-%S") + '.jpg'
    stylized_image.save('outputs/' + file_name)

main()
