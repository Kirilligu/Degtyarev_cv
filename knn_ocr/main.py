import cv2
import numpy as np
import pathlib
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops

def binarize_resize(img,size):
    h, w = size[0] - img.shape[0], size[1]- img.shape[1]
    if h < 0: h = 0
    if w < 0: w = 0
    if img.ndim == 3:
        padded_img = np.pad(img, ((h//2,h-h//2), (w//2,w-w//2), (0, 0)))
    else: padded_img = np.pad(img, ((h//2,h-h//2), (w//2,w-w//2)))
    return (padded_img.mean(axis=2)> 0) if padded_img.ndim ==3 else (padded_img >0)
def load_data(fldr):
    imgs, lbls, cmap, mh, mw = [], [], {}, 0, 0
    for s in fldr.iterdir():
        if s.is_dir():
            ch = s.stem[-1] if len(s.stem) >1 else s.stem
            cmap[ch] = len(cmap) + 1
            for f in s.glob("*.png"):
                i = plt.imread(f )
                i = binarize_resize(i, (mh,mw))
                imgs.append(i)
                lbls.append(cmap[ch])
                mh, mw = max(mh,i.shape[0]), max(mw,i.shape[1])
    return imgs, lbls, cmap, ( mh, mw)
def extract_chars(img, sz):
    r = sorted(regionprops(label(img)),key=lambda x:x.bbox[1])
    ch,pe,g = [],0,[]
    for reg in r:
        l = reg.bbox[1]
        if pe:
            g.append(l- pe)
        pe = reg.bbox[3]
    sp = np.mean(g) + 2*np.std(g) if g else 10
    pe = 0
    for reg in r:
        l = reg.bbox[1]
        if pe and (l - pe)> sp:
            ch.append(None)
        ch.append(binarize_resize(reg.image,sz))
        pe = reg.bbox[3]
    return ch
def train_knn(data, lbls, sz):
    knn = cv2.ml.KNearest_create()
    knn.train(np.array([binarize_resize(i, sz).flatten() for i in data], dtype=np.float32),
              cv2.ml.ROW_SAMPLE, np.array(lbls, dtype=np.float32))
    return knn

base = pathlib.Path(__file__).parent
t_dir = base / "task"
tr_dir = t_dir / "train"
imgs = [plt.imread(str(p)) for p in t_dir.glob("*.png")]
t_data,t_lbls,c_map,m_sz = load_data(tr_dir)
n_map = {v: k for k,v in c_map.items()}
knn = train_knn(t_data, t_lbls, m_sz)
for i,im in enumerate(imgs):
    im = binarize_resize(im,m_sz)
    c = extract_chars(im,m_sz)
    txt = "".join([
        n_map.get(
            int(knn.findNearest(ch.flatten().reshape(1,-1).astype(np.float32),3)[0]), " "
        ) if ch is not None else " "
        for ch in c
    ])
    print(f"{i}) {txt}")
