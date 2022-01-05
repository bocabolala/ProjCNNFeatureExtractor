import numpy as np 
import tensorflow as tf
import tensorflow.keras.applications as models 
# import tensorflow_datasets as tfds
# from libsvm.svmutil import svm_train, svm_problem, svm_parameter

from matplotlib import pyplot



mdls = {}
filters = {}
filters_flat = {}

# Set up models for extract Conv layer
mdls['vgg16'] = models.vgg16.VGG16()
mdls['mob224'] = models.mobilenet.MobileNet()
mdls['inc_v2'] = models.inception_resnet_v2.InceptionResNetV2()
mdls['eff_b2'] = models.efficientnet.EfficientNetB4()
mdls['res50'] = models.resnet50.ResNet50()


def extract_con_kernel(mdls):
    for idx, mdl in enumerate(mdls):
        flag = True 
        layer_idx = 0
        mdl_name = f"{mdl}" 
        layers = mdls[mdl_name].layers[:10]
        # search in first 10 layers for conv layer  
        for layer_idx, layer in enumerate(layers):
            if 'conv' in layer.name and 'pad' not in layer.name:
                flag = False
                break 
            if not flag:
                break
        ker_weight = mdls[mdl_name].layers[layer_idx].get_weights()
        # print(ker_weight)
        if len(ker_weight) == 1:
            filters[mdl_name] = ker_weight[0]
        else:
            filters[mdl_name], _ = ker_weight

        filters_flat[mdl_name] = tf.reduce_mean(filters[mdl_name], axis=2)

        print(mdl_name, 'RGB layer:', layer_idx, layer.name, ' shape', filters[mdl_name].shape)
        print(mdl_name, 'Gray layer:', layer_idx, layer.name, ' shape', filters_flat[mdl_name].shape)
        
    return filters, filters_flat


filters, filters_flat = extract_con_kernel(mdls)

np.save('./filters', filters)
np.save('./filters_flat', filters_flat)




