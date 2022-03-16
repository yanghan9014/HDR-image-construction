import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


image_fns = ['./Memorial_SourceImages/memorial00'+ str(num) +'.png' for num in range(61, 77)]

images = np.asarray([np.asarray(Image.open(fn).convert("RGB")).reshape(-1, 3) for fn in image_fns])
# images = np.transpose(images, (0, 2, 0))
print(images.shape)

valid = np.ones(393216, dtype=np.int8)
for p in images:
  valid = np.logical_and(valid, np.array(p != np.array([0,0,255]))[:,0])
images = images[:, valid, :]
print(images.shape)
# images = images[:, ::1000, :]
num_pixels = 1000
np.random.seed(5487)
images = images[:, np.random.permutation(range(images.shape[1]))[:num_pixels], :]
print(images.shape)

Zmin = np.min(images.reshape(-1, 3), axis=0).astype(np.int16)
Zmax = np.max(images.reshape(-1, 3), axis=0).astype(np.int16)

Zmid = (Zmin + Zmax / 2).astype(np.int16)

# print(Zmin)
# print(Zmax)
# print(Zmid)

def weight(Z, dim):
  if Z <= Zmid[dim]:
    return (Z - Zmin[dim])
  else:
    return (Zmax[dim] - Z) 

Z = images.reshape((-1, 3), order='C')

P = images.shape[0] # Number of pictures
N = images.shape[1] # Number of valid pixels
S = 1 # Smoothness

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
      # B[N*p + n] = (-1) * w * np.log(2)*(p-5)
      B[N*p + n] = (-1) * w * np.log(2)*(p-5)
    A[N * P, Zmid[c]] = 1
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

plt.show()