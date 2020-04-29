import rawpy
import numpy as np
import os

def pack_raw(filename):
    # pack Bayer image to 4 channels
    raw = rawpy.imread('./Sony_ARW/short/' + filename)
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out

def convert_long(filename):
  raw = rawpy.imread('./Sony_ARW/long/' + filename)
  return raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps = 8)

if __name__ == "__main__":
    os.mkdir('Sony_ARW/short_npy')
    os.mkdir('Sony_ARW/long_npy')

    long_filenames = sorted(os.listdir('Sony_ARW/long/'))
    short_filenames = sorted(os.listdir('Sony_ARW/short/'))

    _ = [np.save('./Sony_ARW/long_npy/' + fname[:-4] + '.npy', convert_long(fname)) for fname in tqdm.tqdm(long_filenames)]
    _ = [np.save('./Sony_ARW/short_npy/' + fname[:-4] + '.npy', pack_raw(fname)) for fname in tqdm.tqdm(short_filenames)]
    
    # Copy the two new folders './Sony_ARW/long_npy/' and './Sony_ARW/short_npy/' into the 'Sony' folder present in the root directory.