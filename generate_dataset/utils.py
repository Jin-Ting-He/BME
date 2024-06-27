import numpy as np
import cv2
blur_mag = np.load("disk2/jthe/datasets/GOPRO_blur_magnitude/train/frame15/GOPR0384_11_01/blur_mag_np/010458.npy")
print(blur_mag)
blur_map = blur_mag/100
blur_map = np.uint8(255-(blur_map*255))

cv2.imwrite("home/jthe/BME/blur-magnitude-estimator/generate_dataset/test2.png", blur_map)