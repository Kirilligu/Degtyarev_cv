import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_closing
from skimage.measure import label, regionprops, euler_number
from collections import defaultdict

def identify_symbol(area):
    if area.image.mean() == 1.0:
        return '-'
    closed_image = binary_closing(area.image, structure=np.ones((2, 2)))
    euler_num = euler_number(closed_image, 1)
    if euler_num == -1:
        left_section = area.image[:, :area.image.shape[1] // 2]
        return 'B' if np.sum(np.mean(left_section, axis=0) == 1) > 3 else '8'
    elif euler_num == 0:
        adjusted_image = area.image.copy()
        adjusted_image[-1, :] = 1
        euler_num_adjusted = euler_number(adjusted_image)
        if euler_num_adjusted == -1:
            return 'A'
        vertical_lines = np.sum(np.mean(area.image, axis=0) == 1) > 3
        if not vertical_lines:
            return '0'
        center_y, center_x = area.local_centroid
        return 'P' if abs(center_y / area.image.shape[0] - 0.5) > 0.08 else 'D'
    if np.sum(np.mean(area.image, axis=0) == 1) > 3:
        return '1'
    if area.eccentricity < 0.45:
        return '*'
    padded_image = np.pad(area.image, pad_width=1, mode='maximum')
    padded_euler_num = euler_number(padded_image)
    return '/' if padded_euler_num == -1 else ('X' if padded_euler_num == -3 else 'W')

grayscale_image = plt.imread("symbols.png").mean(2)
binary_image = (grayscale_image > 0).astype(int)
labeled_image = label(binary_image)
symbol_counts = defaultdict(int)
for area in regionprops(labeled_image):
    symbol = identify_symbol(area)
    symbol_counts[symbol] += 1

print(symbol_counts)
