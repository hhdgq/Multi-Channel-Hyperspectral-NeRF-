import numpy as np
import os
import imageio
import re


def read_in_order_npy(file_path):
    pattern = re.compile(r'(\d+)_')
    file_list = [file for file in os.listdir(file_path) if file.endswith('npy')]
    imgfiles = sorted(file_list, key=lambda x: int(pattern.search(x).group(1)))
    return imgfiles


def read_in_order_rgb(file_path):
    pattern = re.compile(r'(\d+)\.')
    file_list = [file for file in os.listdir(file_path) if file.endswith('jpg') or file.endswith('png')]
    imgfiles = sorted(file_list, key=lambda x: int(pattern.search(x).group(1)))
    return imgfiles


def main():
    spectral = False
    bw = False
    filepath_u = 'E:/mrc/papers/NeRF/nerf-pytorch-master/data/nerf_hlfn_npy/mofang1'
    filepath = filepath_u + '/images'
    # imgfiles = [os.path.join(filepath, f) for f in sorted(os.listdir(filepath)) if f.endswith('npy')]
    imgfiles = read_in_order_npy(filepath)
    hlfn_files = []
    for i in range(len(imgfiles)):
        hlfn_file = np.load(filepath + '/' + imgfiles[i])
        hlfn_files.append(hlfn_file)
    hlfn_files_mat = np.reshape(hlfn_files, ([48, hlfn_file.shape[1], hlfn_file.shape[2], hlfn_file.shape[3]]))
    if bw:
        hlfn_files_bw = np.mean(hlfn_files_mat, 1)
    else:
        hlfn_files_rgb = np.stack(
            [np.sum(hlfn_files_mat[:, 19:, ...], axis=1), np.sum(hlfn_files_mat[:, 12:22, ...], axis=1),
             np.sum(hlfn_files_mat[:, :14, ...], axis=1)], axis=3)

    if spectral:
        file_save_path = filepath_u + '/image_cvt_spectral/'
        if not os.path.exists(file_save_path):
            os.mkdir(file_save_path)
    else:
        file_save_path = filepath_u + '/image_cvt_angle/'
        if not os.path.exists(file_save_path):
            os.mkdir(file_save_path)

    if spectral:
        for i in range(hlfn_file.shape[1]):
            imageio.imwrite(file_save_path + 'image_spe_{}'.format(i) + '.jpg', hlfn_file[0, i, ...].astype(np.uint8))
    else:
        for i in range(len(imgfiles)):
            if bw:
                imageio.imwrite(file_save_path + 'image_bw_{}'.format(i) + '.jpg', hlfn_files_bw[i].astype(np.uint8))
            else:
                imageio.imwrite(file_save_path + 'image_rgb_{}'.format(i) + '.jpg',
                                ((hlfn_files_rgb[i] / np.max(hlfn_files_rgb[i])) * 255).astype(np.uint8))


if __name__ == '__main__':
    main()
