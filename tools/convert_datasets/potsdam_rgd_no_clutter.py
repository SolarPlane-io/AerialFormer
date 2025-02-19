import glob
import os
import os.path as osp
import tempfile
import zipfile
import rarfile

import cv2
import mmcv
import numpy as np

from tools.convert_datasets.potsdam_no_clutter import clip_big_image, parse_args

""" 
This prepares an RGD (red-green-depth) raster dataset using the Potsdam orthoimages 
for the red and green channels and DSM elevations (normalized to the range of 0-255)
in place of the usual blue channel. The output file layout in data/potsdam is the
same as what is produced running potsdam_no_clutter.py so once this dataset has been
generated from the raw Potsdam dataset you can run a training job in the same way (by
running python tools/train.py configs/aerialformer/aerialformer_tiny_512x512_5_potsdam.py)
"""

def generate_rgd(rgb_path, dsm_path, dest_path):
    dsm_img = cv2.imread(dsm_path, cv2.IMREAD_UNCHANGED)
    dsm_arr = np.array(dsm_img, dtype=np.uint8)
    min_elev = dsm_arr.min()
    max_elev = dsm_arr.max()
    elev_range = max_elev - min_elev
    step = 255 / elev_range
    zero_floor_arr = dsm_arr - min_elev
    normalized_dsm_arr = zero_floor_arr * step
    ortho_img = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
    ortho_arr = np.array(ortho_img, dtype=np.uint8)
    rgd_arr = ortho_arr
    # There's one DSM in Potsdam that's a pixel smaller in one dimension than its RGB mate. Detect and fix that.
    if rgd_arr.shape[0] != normalized_dsm_arr.shape[0] or rgd_arr.shape[1] != normalized_dsm_arr.shape[1]:
        print(f'RGB raster size {rgd_arr.shape} does not match DSM {normalized_dsm_arr.shape}: resizing DSM')
        new_dim = np.array([rgd_arr.shape[0], rgd_arr.shape[1]])
        normalized_dsm_arr = cv2.resize(normalized_dsm_arr, new_dim, interpolation=cv2.INTER_LINEAR)
    # Substitute the elevation array for the blue channel in the ortho
    # Note: the ortho image channels are ordered BGR.
    rgd_arr[:, :, 0] = normalized_dsm_arr.astype(np.uint8)
    cv2.imwrite(dest_path, rgd_arr)

def main():
    args = parse_args()
    splits = {
        'train': [
            '2_10', '2_11', '2_12', '3_10', '3_11', '3_12', '4_10', '4_11',
            '4_12', '5_10', '5_11', '5_12', '6_10', '6_11', '6_12', '6_7',
            '6_8', '6_9', '7_10', '7_11', '7_12', '7_7', '7_8', '7_9'
        ],
        'val': [
            '5_15', '6_15', '6_13', '3_13', '4_14', '6_14', '5_14', '2_13',
            '4_15', '2_14', '5_13', '4_13', '3_14', '7_13'
        ]
    }

    dataset_path = args.dataset_path
    if args.out_dir is None:
        out_dir = osp.join('data', 'potsdam')
    else:
        out_dir = args.out_dir

    # unpack the DSM files
    rar_list = glob.glob(os.path.join(dataset_path, '*.rar'))
    if len(rar_list) == 0 or "raw_data/Potsdam/1_DSM.rar" not in rar_list:
        print('No 1_DSM.rar file found in path {}'.format(dataset_path))
        return

    # unpack the ortho files
    zipp_list = glob.glob(os.path.join(dataset_path, '*.zip'))
    if len(zipp_list) == 0 or "raw_data/Potsdam/2_Ortho_RGB.zip" not in zipp_list:
        print('No 2_Ortho_RGB.zip file found in path {}'.format(dataset_path))
        return

    if "raw_data/Potsdam/5_Labels_all_noBoundary.zip" not in zipp_list:
        print('No 5_Labels_all_noBoundary.zip file found in path {}'.format(dataset_path))
        return

    mmcv.mkdir_or_exist(osp.join(os.getcwd(), "raw_data/Potsdam/6_RGD"))

    with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmp_dir:
        dsm_rar_file = rarfile.RarFile("raw_data/Potsdam/1_DSM.rar")
        dsm_rar_file.extractall(tmp_dir)
        ortho_zip_file = zipfile.ZipFile("raw_data/Potsdam/2_Ortho_RGB.zip")
        ortho_zip_file.extractall(tmp_dir)
        ortho_src_path_list = glob.glob(os.path.join(f"{tmp_dir}/2_Ortho_RGB", '*.tif'))
        for rgb_path in ortho_src_path_list:
            idx_i, idx_j = osp.basename(rgb_path).split('_')[2:4]
            idx_i_str = f"{idx_i}".zfill(2) # zero pad the index because that's how the dsm files are named
            idx_j_str = f"{idx_j}".zfill(2)
            dsm_path = os.path.join(f"{os.getcwd()}/raw_data/Potsdam/1_DSM", f"dsm_potsdam_{idx_i_str}_{idx_j_str}.tif")
            dest_path = os.path.join(f"{os.getcwd()}/raw_data/Potsdam/6_RGD", f"rgd_potsdam_{idx_i}_{idx_j}_.png")
            generate_rgd(rgb_path, dsm_path, dest_path)

    print('Making directories...')
    mmcv.mkdir_or_exist(osp.join(out_dir, 'img_dir', 'train'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'img_dir', 'val'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'train'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'val'))

    print('Find the data', zipp_list)

    rgd_list = glob.glob(os.path.join("raw_data/Potsdam/6_RGD", '*.png'))
    prog_bar = mmcv.ProgressBar(len(rgd_list))
    for dest_path in rgd_list:
        idx_i, idx_j = osp.basename(dest_path).split('_')[2:4]
        data_type = 'train' if f'{idx_i}_{idx_j}' in splits['train'] else 'val'
        clip_size, stride_size = (args.clip_size, args.stride_size) if data_type == 'train' else (
        args.clip_size, args.clip_size)
        # NOTE: If we overlap the images on testing, it essentially means evaluating the weighted interior of larger images.
        # Therefore, to make a fair comparison of the results, there is no need to overlap the images for testing.
        dst_dir = osp.join(out_dir, 'img_dir', data_type)
        clip_big_image(dest_path, dst_dir, clip_size, stride_size, to_label=False)
        prog_bar.update()

    with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmp_dir:
        labels_zip = zipfile.ZipFile("raw_data/Potsdam/5_Labels_all_noBoundary.zip")
        labels_zip.extractall(tmp_dir)
        labels_list = glob.glob(os.path.join(tmp_dir, '*.tif'))
        prog_bar = mmcv.ProgressBar(len(labels_list))
        for label_path in labels_list:
            idx_i, idx_j = osp.basename(label_path).split('_')[2:4]
            data_type = 'train' if f'{idx_i}_{idx_j}' in splits['train'] else 'val'
            clip_size, stride_size = (args.clip_size, args.stride_size) if data_type == 'train' else (
                args.clip_size, args.clip_size)
            dst_dir = osp.join(out_dir, 'ann_dir', data_type)
            clip_big_image(label_path, dst_dir, clip_size, stride_size, to_label=True)
            prog_bar.update()

    print('Removing the temporary files...')

    print('Done!')


if __name__ == '__main__':
    main()
