import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, binary_erosion, label

def count_star(image):
    dilation_structure = np.ones((1, 2))
    dilated_image = binary_dilation(image, dilation_structure).astype("uint16")
    labeled_dilated_image, num_objects_dilated = label(dilated_image)
    erosion_structure = np.ones((2, 2))
    eroded_image = binary_erosion(image, structure=erosion_structure).astype("uint16")
    _, num_objects_eroded = label(eroded_image)
    return num_objects_dilated - num_objects_eroded

if __name__ == "__main__":
    star_image = np.load("stars.npy")
    star_count = count_star(star_image)

    plt.subplot(1, 2, 1)
    plt.imshow(star_image, cmap='gray')                                                     #исходное

    plt.subplot(1, 2, 2)
    plt.imshow(binary_erosion(star_image, structure=np.ones((2, 2))), cmap='gray')          #обработанное

    print(f"Количество звезд: {star_count}")
    plt.show()
