import numpy as np
import cv2
blur_mag = np.load("disk2/jthe/datasets/GOPRO_blur_magnitude/test/frame11/GOPR0384_11_05/blur_mag_np/044006.npy")
print(blur_mag)
blur_map = blur_mag/np.max(blur_mag)
blur_map = np.uint8(255-(blur_map*255))

cv2.imwrite("home/jthe/BME/blur-magnitude-estimator/generate_dataset/test.png", blur_map)