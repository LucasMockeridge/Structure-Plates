# Code to create Pix2Pix training data

import cv2 
import os
import pandas as pd
import numpy as np

GT_DIR    = "../obc/groundtruth.csv"
EDGES_DIR = "A/"
REAL_DIR  = "B/"
DATA_DIR  = "../obc/data/"

df = pd.read_csv(GT_DIR, header=None, names=["path", "ground"]).replace('"', '', regex=True)
df['path'] = df['path'].apply(os.path.basename)
truth = df.set_index('path')['ground'].to_dict()
train = df['path'].sample(frac=0.8, random_state=1)
test = df['path'].drop(train.index)
train, test = train.tolist(), test.tolist()

def main():
    for file in os.listdir(DATA_DIR):
        img = cv2.imread(DATA_DIR + file, cv2.IMREAD_GRAYSCALE)
        re = cv2.resize(img, (160,256), interpolation=cv2.INTER_CUBIC)
        v = np.median(re)
        lower = int(max(0, (1.0 - 0.33) * v))
        upper = int(min(255, (1.0 + 0.33) * v))
        can = cv2.Canny(re,lower,upper)
        dir = "train" if file in train else "test"
        cv2.imwrite(REAL_DIR + dir + "/" + file, re)
        cv2.imwrite(EDGES_DIR + dir + "/" + file, can)

main()
