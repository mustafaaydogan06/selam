import sys, numpy as np
from PIL import Image, ImageFilter
IMG, OUT = sys.argv[1], sys.argv[2]
shirt = Image.open(f'{IMG}/1.jpg').convert('RGB')
art = Image.open(f'{IMG}/2.jpg').convert('RGB')

# ---------- 1) transparent PNG of the design ----------
a = np.asarray(art).astype(np.float32)/255
h, w = a.shape[:2]
mx = a.max(2)
# near-black sky/shadows -> transparent (blends into a black shirt)
lum_a = np.clip((mx-0.07)/0.22, 0, 1)**0.8
# soft feathered edges (rounded-rect falloff)
yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
d = np.minimum.reduce([xx, w-1-xx, yy, h-1-yy])
f = 0.15*w
edge_a = np.clip(d/f, 0, 1); edge_a = edge_a*edge_a*(3-2*edge_a)
alpha = lum_a*edge_a
# keep the IRON MAIDEN logo crisp even inside the feather zone
r,g,b = a[...,0],a[...,1],a[...,2]
logo = ((r>0.45)&(r>g*1.8)&(r>b*1.8)&(yy<0.21*h)&(xx<0.45*w)).astype(np.uint8)*255
logo = np.asarray(Image.fromarray(logo).filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(1.5))).astype(np.float32)/255
alpha = np.maximum(alpha, logo*lum_a.clip(0.6,1))
alpha_img = Image.fromarray((alpha*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2))
design = art.copy(); design.putalpha(alpha_img)
design.save(f'{OUT}/iron_maiden_tasarim_seffaf.png')

# ---------- 2) remove the old print from the shirt ----------
s = np.asarray(shirt).astype(np.float32)
x0,x1,y0,y1 = 412,848,320,968
reg = s[y0:y1, x0:x1]
mask = (reg.max(2) > 33)
m = Image.fromarray((mask*255).astype(np.uint8)).filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.GaussianBlur(3))
mask = np.asarray(m).astype(np.float32)/255 > 0.05
# diffusion inpaint on a downscaled copy
sc = 4
small = Image.fromarray(reg.astype(np.uint8)).resize(((x1-x0)//sc, (y1-y0)//sc), Image.BILINEAR)
sm = np.asarray(small).astype(np.float32)
mk = np.asarray(Image.fromarray(mask.astype(np.uint8)*255).resize(small.size, Image.BILINEAR)) > 0
fill = sm.copy(); fill[mk] = np.median(sm[~mk], axis=0)
for _ in range(1500):
    p = np.pad(fill, ((1,1),(1,1),(0,0)), mode='edge')
    avg = (p[:-2,1:-1]+p[2:,1:-1]+p[1:-1,:-2]+p[1:-1,2:])/4
    fill[mk] = avg[mk]
big = np.asarray(Image.fromarray(np.clip(fill,0,255).astype(np.uint8)).resize((x1-x0, y1-y0), Image.BICUBIC)).astype(np.float32)
rng = np.random.default_rng(1)
noise = rng.normal(0, 1.6, big.shape[:2])[..., None]
big = big + noise
mb = np.asarray(Image.fromarray(mask.astype(np.uint8)*255).filter(ImageFilter.GaussianBlur(4))).astype(np.float32)[..., None]/255
s[y0:y1, x0:x1] = reg*(1-mb) + big*mb
clean = s.copy()

# ---------- 3) place design: centred, ~10 cm below collar ----------
px_per_cm = 745/52.0           # pit-to-pit ~745 px ≈ 52 cm
collar_y = 203                 # bottom of front neck rib
top = int(collar_y + 10*px_per_cm)
dw = int(30*px_per_cm)         # ~30 cm wide print
dh = int(dw*h/w)
cx = 627
left = cx - dw//2
d = design.resize((dw, dh), Image.LANCZOS)
da = np.asarray(d).astype(np.float32)
rgb, al = da[..., :3], da[..., 3:]/255
# pick up fabric shading/folds so the print sits "in" the shirt
patch = clean[top:top+dh, left:left+dw]
L = patch.mean(2, keepdims=True)
Lb = np.asarray(Image.fromarray(np.clip(L[...,0]*6,0,255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(6))).astype(np.float32)[...,None]/6
shade = np.clip(Lb/np.median(Lb), 0.75, 1.25)
ink = np.clip(rgb*0.95*shade + rng.normal(0, 2.0, rgb.shape[:2])[..., None], 0, 255)
clean[top:top+dh, left:left+dw] = patch*(1-al) + ink*al
res = Image.fromarray(np.clip(clean,0,255).astype(np.uint8))
res.save(f'{OUT}/tisort_mockup_iron_maiden.png')
res.save(f'{OUT}/tisort_mockup_iron_maiden.jpg', quality=95)
print('design', dw, dh, 'at', left, top)
