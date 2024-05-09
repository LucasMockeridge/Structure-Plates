# Code to generate synthetic images using Pillow alone

from PIL import Image, ImageFont, ImageDraw, ImageOps, ImageFilter, ImageEnhance
import numpy as np
import cv2

OUT_DIR  = "out/"
FONT_DIR = "../CharlesWright-Bold.otf"

letters = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
digits = ['0','1','2','3','4','5','6','7','8','9']

w, h = 60, 120
font = ImageFont.truetype(FONT_DIR, 16)

def getFirst():
    n = np.random.randint(1,5)
    str = ''.join(np.random.choice(letters, n))
    if n == 4:
        str = str[:3] + np.random.choice(digits)
    return str

def getSecond():
    n = np.random.randint(1,4)
    str = ''.join(np.random.choice(digits, n))
    return str

def getThird():
    n = np.random.randint(2,5)
    str = ''.join(np.random.choice(digits, n-1))
    if n == 4:
        last = np.random.choice(letters)
    else:
        last = np.random.choice(digits + letters)
    str += last
    return str

def getFourth():
    n = np.random.randint(2,4)
    str = ''.join(np.random.choice(letters, n))
    return str

def getCode():
    n = np.random.randint(3,5)
    str = getFirst() + '\n' + getSecond() + '\n' + getThird() + '\n'
    if n == 4:
        str += getFourth() + '\n'
    return str, n

def newCoords(M, coords):
    new = []
    for (c, x, y, w, h) in coords:
        mat = np.hstack([np.array([[x],[y],[1]]),
                         np.array([[w],[y],[1]]),
                         np.array([[w],[h],[1]]),
                         np.array([[x],[h],[1]])])
        out = M @ mat
        p0, p1 = out[:,0].tolist(), out[:,1].tolist()
        p2, p3 = out[:,2].tolist(), out[:,3].tolist()
        xs = [p0[0][0], p1[0][0], p2[0][0], p3[0][0]]
        ys = [p0[1][0], p1[1][0], p2[1][0], p3[1][0]]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        new.append((c, int(minx), int(miny), int(maxx), int(maxy)))
    return new

def writeCoords(file, coords):
    f = open(file + ".box", "w")
    for (l, x1, y1, x2, y2) in coords:
        a = h-1 - y1
        b = h-1 - y2
        for c in l:
            f.write(f"{c} {int(x1)} {int(b)} {int(x2)} {int(a)} 0\n")
        f.write(f"\t {int(x1)} {int(b)} {int(x2)} {int(a)} 0\n")
    f.close()

def writeGround(file, ground):
    f = open(file + ".gt.txt", "w")
    f.write(ground)
    f.close()

def makeCanvas():
    ground, n = getCode()
    msg = ground[:-1]
    canvas = Image.new('RGB', [w, h], (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    b = h / (2 ** (n-1))
    lines = msg.split('\n')
    coords = []
    x, W = 0, 0
    for i, l in enumerate(lines):
        draw.text((w/2,b+(i*h/4)), l, fill='black', font=font, anchor="mm", align="center")
        xl,yl,wl,hl = draw.textbbox((w/2,b+(i*h/4)), l, font=font, anchor="mm", align="center")
        coords.append((l, xl, yl, wl, hl))
        if wl > W:
            W, x = wl, xl

    # Add lines
    for i in range(len(coords)-1):
        e0, s1 = coords[i][4], coords[i+1][2]
        m = (e0 + s1) / 2
        draw.rectangle((x,m,W,m+1), outline="black")
    return canvas, coords, ground

def transform(canvas):
    y = np.random.randint(-5,6)
    p = np.random.randint(-5,6)
    r = np.random.randint(-5,6)
    tx = np.random.randint(-2,3)
    ty = np.random.randint(-2,3)
    sx = 1 + np.random.uniform(-0.05, 0.05)
    sy = 1 + np.random.uniform(-0.05, 0.05)

    yc, ys = np.cos(np.radians(y)), np.sin(np.radians(y))
    pc, ps = np.cos(np.radians(p)), np.sin(np.radians(p))
    rc, rs = np.cos(np.radians(r)), np.sin(np.radians(r))

    R = np.matrix([[ pc * rc, (ys * ps * rc) - (yc * rs), 0],
                   [ pc * rs, (ys * ps * rs) + (yc * rc), 0],
                   [       0,                          0, 1]])

    T = np.matrix([[ 1, 0, tx],
                   [ 0, 1, ty],
                   [ 0, 0,  1],
                  ], dtype=np.float32)

    S = np.matrix([[ sx,  0, 0],
                   [  0, sy, 0],
                   [  0,  0, 1],
                  ], dtype=np.float32)

    M = T @ R @ S
    out = cv2.warpPerspective(np.array(canvas), M, (w, h), borderValue=(120,120,120))
    return Image.fromarray(out), M

def adjustBrightness(canvas):
    e = np.random.uniform(0.8,1)
    return ImageEnhance.Brightness(canvas).enhance(e)

def addBlur(canvas):
    horiz = ImageFilter.Kernel((3,3), (0, 0, 0, 1, 1, 1, 0, 0, 0))
    return canvas.filter(horiz)

def addNoise(canvas):
    canvas = np.array(canvas)
    noise = np.random.normal(0, 0.2, canvas.shape).astype(np.uint8)
    canvas += noise
    return Image.fromarray(np.clip(canvas, 0, 255))

def checkCoords(coords):
    for (_, x1, y1, x2, y2) in coords:
        if x1 < 0 or x1 >= w or x2 < 0 or x2 >= w or y1 < 0 or y1 >= h or y2 < 0 or y2 >= h:
            return False
    return True

def main():
    i, N = 0, 5
    while i < N:
        im, coords, ground = makeCanvas()
        im, M = transform(im)
        im = addNoise(addBlur(adjustBrightness(im)))
        coords = newCoords(M, coords)
        if checkCoords(coords):
            file = f"{OUT_DIR}/eng_{i}"
            #writeCoords(file, coords)
            #writeGround(file, ground)
            im.save(file + ".tif")
            i += 1
main()
