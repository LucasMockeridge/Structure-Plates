import cv2 
import pytesseract
import os
import pandas as pd
import numpy as np
from PIL import Image

GT_DIR   = "/home/lucas/Documents/groundtruth.csv"
DATA_DIR = "/home/lucas/Documents/data/"

def levenshtein(xs, ys):

    if len(xs) > len(ys):
        xs, ys = ys, xs

    ref = len(xs)
    can = len(ys)

    if ref == 0:
        return can
    if can == 0:
        return ref

    curr = [x for x in range(ref+1)]

    for i in range(1, can+1):
        prev = i
        for j in range(1, ref+1):
            if ys[i-1] == xs[j-1]:
                val = curr[j-1]
            else:
                val = min(curr[j-1] + 1, prev + 1, curr[j] + 1)
            curr[j-1] = prev
            prev = val
        curr[ref] = prev

    return curr[ref]

def cer(can, ref):
    return levenshtein(can, ref) / max(len(ref), len(can))

def main():
    df = pd.read_csv(GT_DIR, header=None, names=["path", "ground"]).replace('"', '', regex=True)
    df['path'] = df['path'].apply(os.path.basename)
    truth = df.set_index('path')['ground'].to_dict()

    avg = 0
    for file in os.listdir(DATA_DIR):
        img = cv2.imread(DATA_DIR + file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        re = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        _, thresh = cv2.threshold(re, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    
        rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        dil = cv2.dilate(thresh, rect_kern, iterations = 1)

        mser = cv2.MSER_create()
        _, boundingBoxes = mser.detectRegions(dil)
        ms = np.zeros(dil.shape, np.uint8)

        for box in boundingBoxes:
            height, width = dil.shape
            x, y, w, h = box;
            if h < 1.2*w or h > height/4 or w < 5 or h < 20 or w > 60:
                continue
            ms[y:y+h, x:x+w] = dil[y:y+h, x:x+w]

        im2 = 255 - ms
        final = cv2.GaussianBlur(im2,(5,5), 0)
        out = pytesseract.image_to_string(Image.fromarray(final), config="--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        t = truth[file].replace('\\n','\n')
        avg += cer(out, t)
    print(f"Accuracy: {(1 - (avg / len(truth))) * 100}%")

main()

