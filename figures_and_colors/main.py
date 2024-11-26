import numpy as np
from skimage.measure import label, regionprops
from skimage.color import rgb2hsv
import matplotlib.pyplot as plt

image = plt.imread('balls_and_rects.png')
image_hsv = rgb2hsv(image)
labeled_image = label(image.mean(axis=2) > 0)
regions = regionprops(labeled_image)
object_hues = [image_hsv[int(region.centroid[0]), int(region.centroid[1]), 0] for region in regions]
unique_hues = [object_hues[0]]
for hue in sorted(object_hues[1:]):
    if hue - unique_hues[-1] > 0.05:
        unique_hues.append(hue)
shape_counts = {hue: {"total": 0, "rectangles": 0, "circles": 0} for hue in unique_hues}

for region, hue in zip(regions, object_hues):
    closest_hue = min(unique_hues, key=lambda h: abs(h - hue))
    shape_counts[closest_hue]["total"] += 1
    if region.extent == 1:
        shape_counts[closest_hue]["rectangles"] += 1
    else:
        shape_counts[closest_hue]["circles"] += 1

total_shapes = 0
for hue, counts in shape_counts.items():
    print(f"Оттенок {hue:.2f}: количество кругов {counts['circles']}, количество прямоугольников {counts['rectangles']}, общее количество {counts['total']}")
    total_shapes += counts["total"]
print(f"Общее количество фигур по оттенкам: {[counts['total'] for counts in shape_counts.values()]}")
print(f"Общее количество фигур : {total_shapes}")
