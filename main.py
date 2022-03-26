import sys, os
from this import d
from cv2 import AlignMTB
import numpy as np

from PIL import Image
from PIL.ExifTags import TAGS

import matplotlib.pyplot as plt
import glob
import shutil

import seaborn as sns; sns.set_theme(style='white')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import MaxNLocator

from alignMTB import AlignMTBImpl


class HDR:
  def __init__(self, path, aligned_path='aligned', sample_num=1000, smoothness=10, dt=0):
    shutil.rmtree(aligned_path, ignore_errors=True)
    try:
      os.mkdir(aligned_path)
    except FileExistsError:
      print("aligned photo directory exists")
    aligner = AlignMTBImpl(path, aligned_path)
    aligner.process()
    self.sample_num = sample_num
    self.smoothness = smoothness

    image_fns = sorted(glob.glob(os.path.join(path, '*.JPG')))
    if len(image_fns) == 0:
      image_fns = sorted(glob.glob(os.path.join(path, '*.png')))
    if len(image_fns) == 0:
      image_fns = sorted(glob.glob(os.path.join(path, '*.jpg')))

    aligned_image_fns = sorted(glob.glob(os.path.join(aligned_path, '*.png')))
    self.raw_images = np.asarray([np.asarray(Image.open(fn).convert("RGB")) for fn in aligned_image_fns])
    print(self.raw_images.shape)

    self.P = self.raw_images.shape[0] # number of pictures
    self.h = self.raw_images.shape[1]
    self.w = self.raw_images.shape[2]
    self.num_pixels = self.h * self.w # number of total pixels
    self.raw_images = self.raw_images.reshape(self.P, -1, 3)

    if dt != 0:
      self.dt = dt
    else:
      self.dt = []
      for fn in image_fns:
        exif = Image.open(fn)._getexif()
        if exif is not None:
          for (tag, value) in exif.items():
            key = TAGS.get(tag, tag)
            if key == 'ExposureTime':
              self.dt.append(value)
        else:
          break
    if len(self.dt) != self.P:
      self.dt = np.exp(np.log(2) * np.arange(5, 5 - self.P, -1, dtype=np.float64))
    print("Exposure time: ", self.dt)
    self.dt = np.array(self.dt, dtype=np.float64)


    exif = Image.open(image_fns[0])._getexif()
    # print("exp: ", exif.items())
    
    # raise
  def main(self):
    self.alignment()
    self.preproccess()
    self.compute_response_curve()
    self.constructHDR()
    self.tonemapping()

  def alignment(self):
    pass

  def preproccess(self):
    valid = np.ones(self.num_pixels, dtype=np.int8)
    for p in self.raw_images:
      valid = np.logical_and(valid, np.array(p != np.array([0,0,255]))[:,0])
    self.images = self.raw_images[:, valid, :]

    np.random.seed(87)
    self.images = self.images[:, np.random.permutation(range(self.images.shape[1]))[:self.sample_num], :]

    self.Zmin = np.min(self.images.reshape(-1, 3), axis=0).astype(np.int16)
    self.Zmax = np.max(self.images.reshape(-1, 3), axis=0).astype(np.int16)
    self.Zmid = (self.Zmin + self.Zmax / 2).astype(np.int16)
  
  def weight(self, Z, color):
    if Z <= self.Zmid[color]:
      return (Z - self.Zmin[color])
    else:
      return (self.Zmax[color] - Z)

  def compute_response_curve(self):
    print("Computing response curve")
    Z = self.images.reshape((-1, 3), order='C')
    N = self.images.shape[1] # Number of valid pixels
    S = self.smoothness

    for c in range(0,3):
      A = np.zeros((N * self.P + 1 + 254, 256 + N), dtype=np.float32)
      B = np.zeros((N * self.P + 1 + 254), dtype = np.float32)
      with np.nditer(self.images[:,:,c], flags=['multi_index'], order='C') as it:
        for i in it:
          p = it.multi_index[0]
          n = it.multi_index[1]
          w = self.weight(i, c)
          A[N*p + n, i] = w
          A[N*p + n, 256+n] = -w
          B[N*p + n] = w * np.log(self.dt[p])
        A[N * self.P, self.Zmid[c]] = 50
        for j in range(254):
          A[N * self.P + 1 + j, j] = S
          A[N * self.P + 1 + j, j + 1] = -2 * S
          A[N * self.P + 1 + j, j + 2] = S
        # print(A.shape)
        # print(B.shape)
        self.x, _, _, _ = np.linalg.lstsq(A, B, rcond=-1)
    # x[255] = x[254]

    lnE = self.x[256:]
    # print(np.expand_dims(lnE, 1).repeat(P, 1).T.shape)
    lndt = np.expand_dims(np.log(self.dt), axis=1).repeat(N, 1)
    # print(dt.shape)

    lnX = np.expand_dims(lnE, 1).repeat(self.P, 1).T + lndt

    # p = np.polyfit(np.reshape(lnX, -1, order='C'), Z[:,0], 4);
    plt.subplot(2,2,c+1)
    color = ['r','g','b']
    plt.plot( Z[:,c], np.reshape(lnX, -1, order='C'), '.', color = color[c], markersize = 1)
    # plt.plot(p)
    plt.plot(range(256), self.x[:256], '.', markersize = 1)
    # print(x[:256])
    
  def g(self, Z):
    return self.x[:256][Z]

  def weight_array(self, Z):
    r = np.where(Z[:,0] <= self.Zmid[0], Z[:,0] - self.Zmin[0] + 1, self.Zmax[0] - Z[:,0] + 1)
    g = np.where(Z[:,1] <= self.Zmid[1], Z[:,1] - self.Zmin[1] + 1, self.Zmax[1] - Z[:,1] + 1)
    b = np.where(Z[:,2] <= self.Zmid[2], Z[:,2] - self.Zmin[2] + 1, self.Zmax[2] - Z[:,2] + 1)
    w = np.column_stack((r,g,b))
    # w = np.where(Z != np.array([0,0,255]), w, np.array([0,0,0]))
    return w

  def constructHDR(self):
    print("Constructing HDR image")
    lnEi = np.zeros((self.num_pixels, 3), dtype=np.float32)
    weight_sum = np.zeros((self.num_pixels, 3), dtype=np.float32)
    for p in range(self.P):
      lnEi += self.weight_array(self.raw_images[p,:,:]) * (self.g(self.raw_images[p,:,:]) - np.log(self.dt[p]))
      weight_sum += self.weight_array(self.raw_images[p,:,:])

    # print("weight_sum: ", np.min(weight_sum))
    lnEi = lnEi / weight_sum
    self.Ei = np.exp(lnEi)
    del self.raw_images
    del self.images

  def tonemapping(self, key=0.7):
    print("Tone mapping")
    # plt.figure(figsize = (20,20))
    intensity = (54*self.Ei[:,0] + 183*self.Ei[:,1] + 19*self.Ei[:,2]) / 256

    sns.set(rc={'figure.figsize':(200,200)})
    intensity_plot = sns.heatmap(intensity.reshape(self.h, self.w), square=True, cmap='Spectral', norm=LogNorm())
    fig = intensity_plot.get_figure()
    fig.savefig("intensity.png") 

    eps = np.finfo(np.float32).eps # smallest positive value for np.float32 (1.1920929e-07)

    Lw_avg = np.exp(np.average(np.log(self.Ei + eps)))
    # Lw_avg = np.average(Lw)

    # print(Lw.max(), Lw.min())
    # print(np.log2(Lw.max()/Lw.min()))
    print("Lw avg:", Lw_avg)
    print("Ei max:", np.max(self.Ei, axis=0))
    print("Ei min:", np.min(self.Ei, axis=0))
    # print(Lw.min())

    # key = 0.4 # control how light or dark
    Lm = self.Ei / Lw_avg * key
    Lwhite = np.max(Lm) # the smallest luminance to be mapped to 1

    Ld = Lm * (1 + Lm / np.power(Lwhite, 2)) / (1+Lm)
    # Ld = np.where(Ld >= 1, 1, Ld)

    # Zd = np.multiply(Ei, Ld) / Ei

    # Z = (Zd - np.min(Zd)) / (np.max(Zd) - np.min(Zd))

    # plt.figure(figsize = (20,20))
    # plt.imshow(Ld.reshape(self.h, self.w, 3))

    import matplotlib
    matplotlib.image.imsave('output.png', Ld.reshape(self.h, self.w, 3))
    # plt.figure()
    # sns.heatmap(Lw.reshape(768, 512), square=True, cmap='Spectral', norm=LogNorm())
    # plt.figure()
    # sns.heatmap(Ld.reshape(768, 512), square=True, cmap='Spectral', norm=LogNorm())

if __name__ == '__main__':
  hdr = HDR('test')
  hdr.main()