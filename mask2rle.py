import os
import re
import cv2
import glob
import time
import json
import fnmatch
import datetime
import numpy as np
from PIL import Image
from pycococreatortools import pycococreatortools


def rgb2masks(label_name):
    # load images
    lbl_id = os.path.split(label_name)[-1].split('.')[0]
    lbl = cv2.imread(label_name, 1)
    # get images info
    h, w = lbl.shape[:2]
    cell_dict = {}
    idx = 0
    white_mask = np.ones((h, w, 3), dtype=np.uint8) * 255
    # sort obj in 2 classes, output to 2 images
    for i in range(h):
        for j in range(w):
            if tuple(lbl[i][j]) in cell_dict or tuple(lbl[i][j]) == (0, 0, 0):
                continue
            cell_dict[tuple(lbl[i][j])] = idx
            mask = (lbl == lbl[i][j]).all(-1)
            # create mask to each class
            cells = np.where(mask[..., None], white_mask, 0)

            mask_name = './cell_data/train/annotations/' + lbl_id + '_cell_' + str(idx) + '.png'
            cv2.imwrite(mask_name, cells)
            idx += 1
 
 
label_dir = './data/MuLV/segmentation'
label_list = glob.glob(os.path.join(label_dir, '*.png'))

start = time.perf_counter()

for idx, label_name in enumerate(label_list):
    inner_timer_start = time.perf_counter()
    print('Converting image', idx)
    rgb2masks(label_name)
    inner_time = time.perf_counter() - inner_timer_start
    print('Image', idx, 'takes', inner_time, 's')

whole_time = time.perf_counter() - start
print('Converting all images takes', whole_time, 'seconds')

 
 
ROOT_DIR = './shapes/train'
IMAGE_DIR = os.path.join(ROOT_DIR, "shapes_train2017")
ANNOTATION_DIR = os.path.join(ROOT_DIR, "annotations")
 
INFO = {
    "description": "Cell Segmentation Dataset",
    "url": "https://github.com/waspinator/pycococreator",
    "version": "0.1.0",
    "year": 2021,
    "contributor": "Peiyu Lu",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}
 
LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]
 
CATEGORIES = [
    {
        'id': 1,
        'name': 'cell_A',
        'supercategory': 'cell',
    },
    {
        'id': 2,
        'name': 'cell_B',
        'supercategory': 'cell',
    }

]
 
 
def filter_for_jpeg(root, files):
    file_types = ['*.jpeg', '*.jpg', '*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    return files
 
 
def filter_for_annotations(root, files, image_filename):
    file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]
    return files
 
 
def main():
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }
 
    image_id = 1
    segmentation_id = 1
 
    # filter for jpeg images
    for root, _, files in os.walk(IMAGE_DIR):
        image_files = filter_for_jpeg(root, files)
 
        # go through each image
        for image_filename in image_files:
            image = Image.open(image_filename)
            image_info = pycococreatortools.create_image_info(
                    image_id, os.path.basename(image_filename), image.size)
            coco_output["images"].append(image_info)
 
            # filter for associated png annotations
            for root, _, files in os.walk(ANNOTATION_DIR):
                annotation_files = filter_for_annotations(root, files, image_filename)
 
                # go through each associated annotation
                for annotation_filename in annotation_files:
 
                    print(annotation_filename)
                    class_id = [x['id'] for x in CATEGORIES if x['name'] in annotation_filename][0]
 
                    category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}
                    binary_mask = np.asarray(Image.open(annotation_filename)
                                             .convert('1')).astype(np.uint8)
 
                    annotation_info = pycococreatortools.create_annotation_info(
                            segmentation_id, image_id, category_info, binary_mask,
                            image.size, tolerance=2)
 
                    if annotation_info is not None:
                        coco_output["annotations"].append(annotation_info)
 
                    segmentation_id = segmentation_id + 1
 
            image_id = image_id + 1
 
    with open('{}/Vironova_Cell_Seg_2021.json'.format(ROOT_DIR), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)
 
 
if __name__ == "__main__":
    main()