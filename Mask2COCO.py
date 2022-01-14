import os
import re
import cv2
import glob
import json
import fnmatch
import numpy as np
from tqdm import tqdm
from PIL import Image
from pycococreatortools import pycococreatortools


# Path to dataset 
ROOT_PATH = './data/MuLV'
# ROOT_PATH = './data/Polio and proteasomes'


# Path name for subfolder, e.g. ./data/fish/image, ./data/fish/label
# IMG_SUB_PATH should be 'image', LABEL_SUB_PATH = 'label' 
IMG_SUB_PATH = 'gray'
LABEL_SUB_PATH = 'segmentation'

# Basic info for COCO format
INFO = {
    "description": "Segmentation Dataset",
    "url": "https://github.com/bocabolala",
    "version": "0.1.0",
    "year": 2021,
    "contributor": "Peiyu Lu",
    "date_created": "Nov, 2021"
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "https://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]
# Provided with 3 categories since the dataset has only 3 
CATEGORIES = [
    {
        'id': 1,
        'name': 'obj_1',
        'supercategory': 'particles',
    },
    {
        'id': 2,
        'name': 'obj_2',
        'supercategory': 'particles',
    },
    {
        'id': 3,
        'name': 'obj_3',
        'supercategory': 'particles',
    },
    {
        'id': 4,
        'name': 'obj_4',
        'supercategory': 'particles',
    }, 
    {
        'id': 5,
        'name': 'obj_5',
        'supercategory': 'particles',
    },

]


def img2instance(label_dir, annotation_dir):
    print('Converting image-level masks to instance-level masks')
    label_list = glob.glob(os.path.join(label_dir, '*.png'))

    assert len(label_list)!=0, 'Error: No image in path'

    for idx, label_name in enumerate(tqdm(label_list)):
        os.makedirs(annotation_dir, exist_ok=True)
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
            temp = np.zeros((h, w), dtype=np.uint8)
            # print('idx', idx, 'color', color)
            color_idx[f'{color}'] = np.where(label == color)
            # print(color_idx[f'{color}'])
            temp[color_idx[f'{color}']] = 255
            ret, binary = cv2.threshold(temp, 127, 255, cv2.THRESH_BINARY)
            ret, connected_components_labels = cv2.connectedComponents(binary, 8)

            for idx2 in np.unique(connected_components_labels):
                if idx2 == 0:
                    continue  # skip background with index == 0
                single_cc = np.zeros((h, w), dtype=np.uint8)
                cc_idx = np.where(connected_components_labels == idx2)
                single_cc[cc_idx] = 255

                mask_name = label_id + '_obj_' + str(idx) + '_part_' + str(idx2) + '.png'

                cv2.imwrite(os.path.join(annotation_dir, mask_name), single_cc)
                # cv2.imshow('123', single_cc)
                # cv2.waitKey(0)


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


def coco_json_creator(root_dir, image_dir, annotation_dir):
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1
    segmentation_id = 1

    print('Creating COCO json over instance-level masks')
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

    # filter for jpeg images
    for root, _, files in os.walk(image_dir):
        image_files = filter_for_jpeg(root, files)

        # go through each image
        for image_filename in tqdm(image_files):
            image = Image.open(image_filename)
            image_info = pycococreatortools.create_image_info(
                image_id, os.path.basename(image_filename), image.size)
            coco_output["images"].append(image_info)

            # filter for associated png annotations
            for root, _, files in os.walk(annotation_dir):
                annotation_files = filter_for_annotations(root, files, image_filename)

                # go through each associated annotation
                for annotation_filename in annotation_files:
                    # check class depending name
                    class_id = [x['id'] for x in CATEGORIES if x['name'] in annotation_filename][0]
                    category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}
                    binary_mask = np.asarray(Image.open(annotation_filename)
                                             .convert('1')).astype(np.uint8)
                    # create annotation info
                    annotation_info = pycococreatortools.create_annotation_info(
                        segmentation_id, image_id, category_info, binary_mask,
                        image.size, tolerance=2)

                    # print('anno_info',annotation_info)
                    if annotation_info is not None:
                        coco_output["annotations"].append(annotation_info)

                    segmentation_id +=1

            image_id +=1

    with open('{}/instance_train2021.json'.format(root_dir), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)


if __name__ == "__main__":
    
    image_path = os.path.join(ROOT_PATH, IMG_SUB_PATH)
    annotation_path = os.path.join(ROOT_PATH, "annotations")
    label_path = os.path.join(ROOT_PATH, LABEL_SUB_PATH)
    img2instance(label_path, annotation_path)
    coco_json_creator(ROOT_PATH, image_path, annotation_path)

        

