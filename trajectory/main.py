import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops

def plot_tracks(folder="out", frames=100):
    paths=[]
    for t in range(frames):
        points= [obj.centroid for obj in regionprops(label(np.load(f"{folder}/h_{t}.npy")))]
        if not paths:
            paths= [[p ] for p in points]
        else:
            for path in paths:
                path.append(min(points,key=lambda p:np.linalg.norm(np.array(path[-1])-p)))

    [plt.plot(*zip(*path)) for path in paths]
    plt.show()
if __name__ == "__main__":
    plot_tracks()
