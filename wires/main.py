import numpy as np
from scipy import ndimage

def analyze(image):
    binary = (image > 0).astype(np.uint8)
    labeled, total = ndimage.label(binary)
    statuses = []
    for wire_id in range(1, total + 1):
        wire = (labeled == wire_id)
        eroded = ndimage.binary_erosion(wire, structure=np.ones((3, 3)))
        labeled_parts, part_count = ndimage.label(eroded)
        if part_count == 0:
            statuses.append((wire_id, "весь порван"))
        elif part_count == 1:
            statuses.append((wire_id, "целый"))
        else:
            statuses.append((wire_id, f"разорван на {part_count} часть(и)"))
    return total, statuses

for i in range(1, 7):
    img = np.load(f"wires{i}npy.txt")
    total_wires, wire_info = analyze(img)
    print(f"Файл {i}")
    print(f"Количество проводов: {total_wires}")
    for wire_id, status in wire_info:
        print(f"Провод {wire_id} {status}")
    print()
