import os
import cv2
import numpy as np

if __name__ == "__main__":

    nb_images = 10000

    uniform_noise_dataset_directory = os.path.join("data", "uniform_noise")
    images = np.random.uniform(size=(nb_images, 32, 32, 3)) * 255
    for j in range(nb_images):
        cv2.imwrite(os.path.join(uniform_noise_dataset_directory, f'{j}.jpg'), images[j])

    gaussian_noise_dataset_directory = os.path.join("data", "gaussian_noise")
    images = np.random.randn(nb_images, 32, 32, 3) * 255
    for j in range(nb_images):
        cv2.imwrite(os.path.join(gaussian_noise_dataset_directory, f'{j}.jpg'), images[j])