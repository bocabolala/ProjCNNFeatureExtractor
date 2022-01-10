# Vironova Segmentation

## TO BUILD COSTUMED MASK R-CNN MODEL

### 1. Run Mask2RLE to convert mask annotation to coco RLE(Run-Length Encoding) format annotation

### 2. Run Jupyter Notebook mrcnn_train.ipynb, to generate and train the model 

### 3. Run kernel_extractor.ipynb to extract kernel from the models. (Recommend the weights of conv1_conv in stage 1 of ResNet backbone)

## TO USE KERNEL FROM PRETRAIN MODELS

### Run layer_ext.py and costum your models.