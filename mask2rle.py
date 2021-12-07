import os
import re
import cv2
import glob
import time
import json
import fnmatch
import datetime
import numpy as np
from tqdm import tqdm
from PIL import Image

from pycococreatortools import pycococreatortools


def rgb2masks(label_name):
    # load images
    label_id = os.path.split(label_name)[-1].split('.')[0]
    label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)
    # get images info
 
    h, w = label.shape[:2]

    color_idx = {}

    cc_idx = {}


    # label_color, coler_count = np.unique(label,return_counts=True) 
    # print(label_color, coler_count)

    label_color = np.unique(label)
    # print(label_color)

    for idx, color in enumerate(label_color):
        if idx == 0 and color == 0:
            continue
        temp = np.zeros((h,w),dtype=np.uint8)
        # print('idx', idx, 'color', color)
        color_idx[f'{color}'] = np.where(label == color)
        # print(color_idx[f'{color}'])

        temp[color_idx[f'{color}']] = 255
        ret, binary = cv2.threshold(temp,127,255,cv2.THRESH_BINARY)
        ret2, connected_conponents_labels = cv2.connectedComponents(binary, 8)

        for idx2 in np.unique(connected_conponents_labels):
            if idx2 == 0:
                continue
            single_cc = np.zeros((h, w),dtype=np.uint8)
            cc_idx = np.where(connected_conponents_labels == idx2)
            single_cc[cc_idx] = 255 

            mask_name = './cell_data/train/annotations/'+ label_id + '_obj_' + str(idx) + '_part_' + str(idx2) + '.png'
            cv2.imwrite(mask_name, single_cc)
            # cv2.imshow('123', single_cc)
            # cv2.waitKey(0)
    
 
label_dir = './data/MuLV/segmentation'
label_list = glob.glob(os.path.join(label_dir, '*.png'))

start = time.perf_counter()
for idx, label_name in enumerate(tqdm(label_list)):
    inner_timer_start = time.perf_counter()
    rgb2masks(label_name)
    inner_time = time.perf_counter() - inner_timer_start

print(f"Converting all images takes {time.perf_counter() - start:.3f} seconds")

 
 
# ROOT_DIR = './shapes/train'
# IMAGE_DIR = os.path.join(ROOT_DIR, "shapes_train2017")
# ANNOTATION_DIR = os.path.join(ROOT_DIR, "annotations")
 
# INFO = {
#     "description": "Cell Segmentation Dataset",
#     "url": "https://github.com/bocabolala",
#     "version": "0.1.0",
#     "year": 2021,
#     "contributor": "Peiyu Lu",
#     "date_created": datetime.datetime.utcnow().isoformat(' ')
# }
 
# LICENSES = [
#     {
#         "id": 1,
#         "name": "Attribution-NonCommercial-ShareAlike License",
#         "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
#     }
# ]
 
# CATEGORIES = [
#     {
#         'id': 1,
#         'name': 'cell_A',
#         'supercategory': 'cell',
#     },
#     {
#         'id': 2,
#         'name': 'cell_B',
#         'supercategory': 'cell',
#     }

# ]
 
 
# def filter_for_jpeg(root, files):
#     file_types = ['*.jpeg', '*.jpg', '*.png']
#     file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
#     files = [os.path.join(root, f) for f in files]
#     files = [f for f in files if re.match(file_types, f)]
#     return files
 
 
# def filter_for_annotations(root, files, image_filename):
#     file_types = ['*.png']
#     file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
#     basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
#     file_name_prefix = basename_no_extension + '.*'
#     files = [os.path.join(root, f) for f in files]
#     files = [f for f in files if re.match(file_types, f)]
#     files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]
#     return files
 
 
# def main():
#     coco_output = {
#         "info": INFO,
#         "licenses": LICENSES,
#         "categories": CATEGORIES,
#         "images": [],
#         "annotations": []
#     }
 
#     image_id = 1
#     segmentation_id = 1
 
#     # filter for jpeg images
#     for root, _, files in os.walk(IMAGE_DIR):
#         image_files = filter_for_jpeg(root, files)
 
#         # go through each image
#         for image_filename in image_files:
#             image = Image.open(image_filename)
#             image_info = pycococreatortools.create_image_info(
#                     image_id, os.path.basename(image_filename), image.size)
#             coco_output["images"].append(image_info)
 
#             # filter for associated png annotations
#             for root, _, files in os.walk(ANNOTATION_DIR):
#                 annotation_files = filter_for_annotations(root, files, image_filename)
 
#                 # go through each associated annotation
#                 for annotation_filename in annotation_files:
 
#                     print(annotation_filename)
#                     class_id = [x['id'] for x in CATEGORIES if x['name'] in annotation_filename][0]
 
#                     category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}
#                     binary_mask = np.asarray(Image.open(annotation_filename)
#                                              .convert('1')).astype(np.uint8)
 
#                     annotation_info = pycococreatortools.create_annotation_info(
#                             segmentation_id, image_id, category_info, binary_mask,
#                             image.size, tolerance=2)
 
#                     if annotation_info is not None:
#                         coco_output["annotations"].append(annotation_info)
 
#                     segmentation_id = segmentation_id + 1
 
#             image_id = image_id + 1
 
#     with open('{}/Vironova_Cell_Seg_2021.json'.format(ROOT_DIR), 'w') as output_json_file:
#         json.dump(coco_output, output_json_file)
 
 
# if __name__ == "__main__":
#     main()