import sys
import numpy as np

from PIL import Image
from PIL.ExifTags import TAGS

import matplotlib.pyplot as plt


image_fns = ['./test/B43A59'+ str(num) +'.JPG' for num in range(22, 32)]


raw_images = np.asarray([np.asarray(Image.open(fn).convert("RGB")).reshape(-1, 3) for fn in image_fns])
print(raw_images.shape)

exif = Image.open(image_fns[0])._getexif()
print(exif.ExposureTime)
if exif is not None:
    for (tag, value) in exif.items():
	    key = TAGS.get(tag, tag)
	    print(key + ' = ' + str(value))

raise
valid = np.ones(393216, dtype=np.int8)
for p in raw_images:
  valid = np.logical_and(valid, np.array(p != np.array([0,0,255]))[:,0])
images = raw_images[:, valid, :]
print(images.shape)
# images = images[:, ::1000, :]
num_pixels = 1000
np.random.seed(87)
images = images[:, np.random.permutation(range(images.shape[1]))[:num_pixels], :]
print(images.shape)

Zmin = np.min(images.reshape(-1, 3), axis=0).astype(np.int16)
Zmax = np.max(images.reshape(-1, 3), axis=0).astype(np.int16)

Zmid = (Zmin + Zmax / 2).astype(np.int16)

# print(Zmin)
# print(Zmax)
# print(Zmid)

def weight(Z, color):
  if Z <= Zmid[color]:
    return (Z - Zmin[color])
  else:
    return (Zmax[color] - Z)

Z = images.reshape((-1, 3), order='C')

P = images.shape[0] # Number of pictures
N = images.shape[1] # Number of valid pixels
S = 10 # Smoothness

for c in range(0,3):
  A = np.zeros((N * P + 1 + 254, 256 + N), dtype=np.float32)
  B = np.zeros((N * P + 1 + 254), dtype = np.float32)
  with np.nditer(images[:,:,c], flags=['multi_index'], order='C') as it:
    for i in it:
      p = it.multi_index[0]
      n = it.multi_index[1]
      w = weight(i, c)
      A[N*p + n, i] = w
      A[N*p + n, 256+n] = -w
      B[N*p + n] = w * np.log(2)*(5-p)
    A[N * P, Zmid[c]] = 50
    for j in range(254):
      A[N * P + 1 + j, j] = S
      A[N * P + 1 + j, j + 1] = -2 * S
      A[N * P + 1 + j, j + 2] = S
    # print(A.shape)
    # print(B.shape)
    x, _, _, _ = np.linalg.lstsq(A, B)
    # x[255] = x[254]

    lnE = x[256:]
    # print(np.expand_dims(lnE, 1).repeat(P, 1).T.shape)
    lndt = np.log(2) * np.expand_dims(np.arange(5, -11, -1), 1).repeat(N, 1)
    # print(dt.shape)

    lnX = np.expand_dims(lnE, 1).repeat(P, 1).T + lndt

    # p = np.polyfit(np.reshape(lnX, -1, order='C'), Z[:,0], 4);
    plt.subplot(2,2,c+1)
    color = ['r','g','b']
    plt.plot( Z[:,c], np.reshape(lnX, -1, order='C'), '.', color = color[c], markersize = 1)
    # plt.plot(p)
    plt.plot(range(256), x[:256], '.', markersize = 1)
    # print(x[:256])
def g(Z):
  return x[:256][Z]

def weight_array(Z):
  r = np.where(Z[:,0] <= Zmid[0], Z[:,0] - Zmin[0], Zmax[0] - Z[:,0])
  g = np.where(Z[:,1] <= Zmid[1], Z[:,1] - Zmin[1], Zmax[1] - Z[:,1])
  b = np.where(Z[:,2] <= Zmid[2], Z[:,2] - Zmin[2], Zmax[2] - Z[:,2])
  w = np.column_stack((r,g,b))
  w = np.where(Z != np.array([0,0,255]), w, np.array([0,0,0]))
  return w

lnEi = np.zeros((768*512, 3), dtype=np.float32)
weight_sum = np.zeros((768*512, 3), dtype=np.float32)
for p in range(16):
  lnEi += weight_array(raw_images[p,:,:]) * (g(raw_images[p,:,:]) - np.log(2)*(5-p))
  weight_sum += weight_array(raw_images[p,:,:])
lnEi = lnEi / weight_sum
Ei = np.exp(lnEi)

import seaborn as sns; sns.set_theme(style='white')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import MaxNLocator


# plt.figure(figsize = (20,20))
intensity = (54*Ei[:,0] + 183*Ei[:,1] + 19*Ei[:,2]) / 256
# sns.heatmap(intensity.reshape(768, 512), square=True, cmap='Spectral', norm=LogNorm())


eps = np.finfo(np.float32).eps # smallest positive value for np.float32 (1.1920929e-07)
# Lw = (54*Ei[:,0] + 183*Ei[:,1] + 19*Ei[:,2]) / 256

Lw_avg = np.exp(np.average(np.log(Ei + eps)))
# Lw_avg = np.average(Lw)

# print(Lw.max(), Lw.min())
# print(np.log2(Lw.max()/Lw.min()))
print("Lw avg:", Lw_avg)
print("Ei max:", np.max(Ei, axis=0))
print("Ei min:", np.min(Ei, axis=0))
# print(Lw.min())

key = 0.4 # control how light or dark
Lm = Ei / Lw_avg * key
Lwhite = np.max(Lm) # the smallest luminance to be mapped to 1

Ld = Lm * (1 + Lm / np.power(Lwhite, 2)) / (1+Lm)
# Ld = np.where(Ld >= 1, 1, Ld)

# Zd = np.multiply(Ei, Ld) / Ei

# Z = (Zd - np.min(Zd)) / (np.max(Zd) - np.min(Zd))

plt.figure(figsize = (20,20))
plt.imshow(Ld.reshape(768, 512, 3))

import matplotlib
matplotlib.image.imsave('tone_mapped_key18e-2.png', Ld.reshape(768,512,3))

# plt.figure()
# sns.heatmap(Lw.reshape(768, 512), square=True, cmap='Spectral', norm=LogNorm())
# plt.figure()
# sns.heatmap(Ld.reshape(768, 512), square=True, cmap='Spectral', norm=LogNorm())