import numpy as np
import os
import cv2

image_path = 'input/content/avril.jpg'

folder, filename = os.path.split(image_path)

filename = filename.replace('.jpg','_noisy.jpg')

def gaussian_noise(img, mean=0, sigma=0.2):
    
    # int -> float (標準化)
    img = img / 255
    # 隨機生成高斯 noise (float + float)
    noise = np.random.normal(mean, sigma, img.shape)
    # noise + 原圖
    gaussian_out = img + noise
    # 所有值必須介於 0~1 之間，超過1 = 1，小於0 = 0
    gaussian_out = np.clip(gaussian_out, 0, 1)
    
    # 原圖: float -> int (0~1 -> 0~255)
    gaussian_out = np.uint8(gaussian_out*255)
    # noise: float -> int (0~1 -> 0~255)
    noise = np.uint8(noise*255)
    
    return gaussian_out

img = cv2.imread(image_path)

img_noisy = gaussian_noise(img)

cv2.imwrite(filename,img_noisy)