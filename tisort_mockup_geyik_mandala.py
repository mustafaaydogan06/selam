import sys, numpy as np
from PIL import Image, ImageFilter, ImageDraw
from scipy import ndimage as nd
IMG, OUT = sys.argv[1], sys.argv[2]
art = Image.open(f'{IMG}/5.jpg').convert('RGB')
a = np.asarray(art).astype(np.float32)
h, w = a.shape[:2]
yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

# ---- 1) cut-out: keep mandala disc + deer + antlers, drop the teal pattern bg ----
# mandala disc (+ small top medallion and light beam), soft edge
disc = np.clip((322 - np.hypot(xx-368, yy-428))/14 + 0.5, 0, 1)
medal = np.clip((36 - np.hypot(xx-367, yy-30))/6 + 0.5, 0, 1)
beam = ((np.abs(xx-367) < 22) & (yy < 140)).astype(np.float32)
# deer head/neck/chest below the disc
body = Image.new('L', (w, h), 0)
ImageDraw.Draw(body).polygon([(150, 470), (215, 560), (205, 690), (210, 760), (205, 830), (218, 900),
    (252, 960), (300, 1005), (368, 1030), (436, 1005), (486, 960), (518, 900), (530, 830),
    (526, 760), (528, 690), (520, 560), (590, 470)], fill=255)
body = nd.gaussian_filter(np.asarray(body).astype(np.float32)/255, 5)
# antlers: warm (brown/tan) pixels connected to the disc
r, g, b = a[..., 0], a[..., 1], a[..., 2]
warm = (r > b+18) & (r >= g-4) & (a.max(2) > 45)
warm = nd.binary_closing(warm, iterations=2)
lab, n = nd.label(warm, structure=np.ones((3, 3)))
sizes = nd.sum(np.ones_like(lab), lab, range(1, n+1))
touch = set(np.unique(lab[(disc > 0.5) & (lab > 0)]))
keep = [i for i in range(1, n+1) if sizes[i-1] > 400 and i in touch]
ant = np.isin(lab, keep) & (yy < 420)
ant = nd.binary_opening(ant, iterations=1)
ant = nd.gaussian_filter(nd.binary_dilation(ant, iterations=1).astype(np.float32), 1.2)
alpha = np.clip(np.maximum.reduce([disc, medal, beam, body, ant]), 0, 1)
# very dark gaps become see-through so the black shirt shows
alpha *= np.clip((a.max(2) - 12)/25, 0, 1)**0.7*0.6 + 0.4*(a.max(2) > 12)
e = np.clip(np.minimum.reduce([xx, w-1-xx, yy, h-1-yy])/(0.04*w), 0, 1)
alpha *= e*e*(3-2*e)
design = Image.fromarray(np.dstack([a, alpha*255]).astype(np.uint8), 'RGBA')
design.save(f'{OUT}/geyik_mandala_tasarim_seffaf.png')

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
res.save(f'{OUT}/tisort_mockup_geyik_mandala.png')
res.save(f'{OUT}/tisort_mockup_geyik_mandala.jpg', quality=95)
print('design', dw, dh, 'at', left, top)
