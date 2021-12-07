import tensorflow.keras.applications as models 
import tensorflow as tf
import numpy as np 

from matplotlib import pyplot
# from tensorflow_examples.models.pix2pix import pix2pixd
# import tensorflow_datasets as tfds
from libsvm.svmutil import svm_train, svm_problem, svm_parameter

# resnet152 = tf.keras.applications.resnet.ResNet152(
#     include_top=True, weights='imagenet', input_tensor=None,
#     input_shape=None, pooling=None, classes=1000,
# )


# densenet121 = tf.keras.applications.densenet.DenseNet121(
#     include_top=True, weights='imagenet', input_tensor=None,
#     input_shape=None, pooling=None, classes=1000
# )


# extract feature conv layers in ResNet
# for i, layer in enumerate(resnet152.layers[:20]):
#     print(i,'\t',layer.trainable,'\t  :',layer.name)

# extract feature conv layers in ResNet



# for i, layer in enumerate(densenet121.layers[:20]):
#     if 'conv' not in layer.name:
#         continue
#     print(layer.name, 'index', i)

mdls = {}
filters = {}
biases = {}

# Set up models for extract Conv layer
mdls['vgg16'] = models.vgg16.VGG16()
mdls['mob224'] = models.mobilenet.MobileNet()
mdls['inc_v2'] = models.inception_resnet_v2.InceptionResNetV2()
mdls['eff_b2'] = models.efficientnet.EfficientNetB4()
mdls['res50'] = models.resnet50.ResNet50()


def extract_con_kernel(mdls):
    for idx, mdl in enumerate(mdls):
        flag = True 
        l_idx = 0
        mdl_name = f"{mdl}" 
        layers = mdls[mdl_name].layers[:10]
        # search in first 10 layers for conv layer  
        for l_idx, layer in enumerate(layers):
            if 'conv' in layer.name and 'pad' not in layer.name:
                flag = False
                break 
            if not flag:
                break
        ker_weight = mdls[mdl_name].layers[l_idx].get_weights()
        # print(ker_weight)
        if len(ker_weight) == 1:
            filters[mdl_name] = ker_weight[0]
        else:
            filters[mdl_name], _ = ker_weight
        print(mdl_name, ' layer:', l_idx, layer.name, ' shape', filters[mdl_name].shape)
        
    return filters


filters = extract_con_kernel(mdls)
np.load('./filters',filters)

# vgg16_filters, vgg16_biases = mdls['vgg16'].layers[1].get_weights()





# resnet152_conv_feature = resnet152.layers[2].get_weights   # get conv1 conv weight in resnet
# densenet121_conv_feature = densenet121.layers[2].get_weights  # get conv1-conv weight in densenet

# print(resnet152.summary())
# print(densenet121.summary())



