import numpy as np
from scipy.ndimage import binary_erosion, label, binary_hit_or_miss

def count_objects(image, structure):
  eroded = binary_erosion(image, structure=structure)
  labeled_array, num_features = label(eroded)
  return num_features

def main():
  image = np.load("ps.npy.txt").astype('uint16')
  print("Общее количество:", np.max(label(image)[0]))
  structsg = np.ones((3, 4, 6)).astype("uint16")
  structsv = np.ones((3, 6, 4)).astype("uint16")

  for h in range(2):
    structsg[h + 1, h * 2:h * 2 + 2, 2:4] = 0
    structsv[h + 1, 2:4, h * 2:h * 2 + 2] = 0
  for i in range(structsg.shape[0]):
    print(structsg[i, :, :])
    result = binary_hit_or_miss(image, structsg[i, :, :])
    print(f"{np.sum(result)}")
  for i in range(structsv.shape[0]):
    print(structsv[i, :, :])
    result = binary_hit_or_miss(image, structsv[i, :, :])
    print(f"{np.sum(result)}")

if __name__ == "__main__":
  main()
