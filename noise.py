import numpy as np
import os
import cv2
import argparse

parser = argparse.ArgumentParser(description='Add noise to all images in a directory')
parser.add_argument('--image_dir', default='input/style',
                    help='Directory of images')

args = parser.parse_args()

image_dir = args.image_dir

for filename in os.listdir(image_dir):
    if not filename.endswith('.jpg') or 'noisy' in filename:
        continue
    image_path = os.path.join(image_dir, filename)
    output_path = image_path.replace('.jpg','_noisy.jpg')

    def gaussian_noise(img, mean=0, sigma=0.3):
        img = img / 255
        noise = np.random.normal(mean, sigma, img.shape)
        gaussian_out = img + noise
        gaussian_out = np.clip(gaussian_out, 0, 1)
        gaussian_out = np.uint8(gaussian_out*255)
        noise = np.uint8(noise*255)
        return gaussian_out

    img = cv2.imread(image_path)

    img_noisy = gaussian_noise(img)

    cv2.imwrite(output_path,img_noisy)