import sys, numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage as nd
IMG, OUT = sys.argv[1], sys.argv[2]
art = Image.open(f'{IMG}/3.jpg').convert('RGB')
a = np.asarray(art).astype(np.float32)
h, w = a.shape[:2]
BG = 229.0
dist = np.abs(a-BG).max(2)

# ---- 1) background matte (flat light-grey bg + enclosed bg holes) ----
lab, n = nd.label(dist < 16)
keep = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
lab2, n2 = nd.label(dist <= 4)
sizes = nd.sum(np.ones_like(lab2), lab2, range(1, n2+1))
for i, s in enumerate(sizes, 1):
    if s > 100:
        keep |= set(np.unique(lab[lab2 == i]))
keep.discard(0)
bg = np.isin(lab, list(keep))
# soft alpha inside the bg region (anti-aliased edges), solid elsewhere
soft = np.clip((dist-3)/13, 0, 1)
alpha = np.where(bg, soft, 1.0)
alpha = np.asarray(Image.fromarray((alpha*255).astype(np.uint8)).filter(ImageFilter.MinFilter(3))
                   .filter(ImageFilter.GaussianBlur(1.0))).astype(np.float32)/255
# decontaminate: remove grey bg spill from semi-transparent edge pixels
am = np.clip(alpha, 0.05, 1)[..., None]
rgb = np.clip((a - (1-am)*BG)/am, 0, 255)
rgb = np.where(alpha[..., None] > 0.98, a, rgb)
# soften the canvas border too (jacket / spades are cut off at the image edge)
yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
d = np.minimum.reduce([xx, w-1-xx, yy, h-1-yy])
e = np.clip(d/(0.06*w), 0, 1); e = e*e*(3-2*e)
alpha = alpha*e
design = Image.fromarray(np.dstack([rgb, alpha*255]).astype(np.uint8), 'RGBA')
design.save(f'{OUT}/mad_hatter_tasarim_seffaf.png')

# ---- 2) clean black shirt (old print removed) ----
shirt = Image.open(f'{IMG}/1.jpg').convert('RGB')
s = np.asarray(shirt).astype(np.float32)
x0, x1, y0, y1 = 412, 848, 320, 968
reg = s[y0:y1, x0:x1]
# whole old print area (incl. its faint dark moon / branches) + any bright leftovers
yy2, xx2 = np.mgrid[0:y1-y0, 0:x1-x0]
mask = ((xx2 > 445-x0) & (xx2 < 820-x0) & (yy2 > 328-y0) & (yy2 < 945-y0)) | (reg.max(2) > 33)
mask = nd.binary_dilation(mask, iterations=4)
k = 8
sm = np.asarray(Image.fromarray(reg.astype(np.uint8)).resize(((x1-x0)//k, (y1-y0)//k), Image.BOX)).astype(np.float32)
small_m = np.asarray(Image.fromarray(mask.astype(np.uint8)*255).resize(((x1-x0)//k, (y1-y0)//k), Image.BOX)) > 0
sm[small_m] = np.median(sm[~small_m], axis=0)
for _ in range(6000):
    p = np.pad(sm, ((1, 1), (1, 1), (0, 0)), mode='edge')
    sm[small_m] = ((p[:-2, 1:-1]+p[2:, 1:-1]+p[1:-1, :-2]+p[1:-1, 2:])/4)[small_m]
big = np.asarray(Image.fromarray(np.clip(sm*8, 0, 255).astype(np.uint8)).resize((x1-x0, y1-y0), Image.BICUBIC)).astype(np.float32)/8
big = nd.gaussian_filter(big, (6, 6, 0))
rng = np.random.default_rng(1)
big += rng.normal(0, 1.5, big.shape[:2])[..., None]
mb = nd.gaussian_filter(mask.astype(np.float32), 6)[..., None]
s[y0:y1, x0:x1] = reg*(1-mb) + big*mb

# ---- 3) place: centred, ~10 cm below collar ----
px_per_cm = 745/52.0
top = int(203 + 10*px_per_cm)
dh = int(42*px_per_cm)            # ~42 cm tall print
dw = int(dh*w/h)                  # ~23.6 cm wide
left = 627 - dw//2
dd = np.asarray(design.resize((dw, dh), Image.LANCZOS)).astype(np.float32)
rgbd, al = dd[..., :3], dd[..., 3:]/255
patch = s[top:top+dh, left:left+dw]
L = nd.gaussian_filter(patch.mean(2), 6)[..., None]
shade = np.clip(L/np.median(L), 0.8, 1.2)
ink = np.clip(rgbd*0.95*shade + rng.normal(0, 2.0, rgbd.shape[:2])[..., None], 0, 255)
s[top:top+dh, left:left+dw] = patch*(1-al) + ink*al
res = Image.fromarray(np.clip(s, 0, 255).astype(np.uint8))
res.save(f'{OUT}/tisort_mockup_mad_hatter.png')
res.save(f'{OUT}/tisort_mockup_mad_hatter.jpg', quality=95)
print('design', dw, dh, 'at', left, top, 'bg frac', bg.mean())
