import cv2
import os
import time
import uuid
from object_detection.utils import dataset_util, label_map_util

WORKSPACE_PATH = 'Tensorflow/workspace'
SCRIPTS_PATH = 'Tensorflow/scripts'
APIMODEL_PATH = 'Tensorflow/models'
ANNOTATION_PATH = WORKSPACE_PATH + '/annotations'
IMAGE_PATH = WORKSPACE_PATH + '/images'
MODEL_PATH = WORKSPACE_PATH + '/models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH + '/pre-trained-models'
CONFIG_PATH = MODEL_PATH + '/my_ssd_mobnet/pipeline.config'
CHECKPOINT_PATH = MODEL_PATH + '/my_ssd_mobnet/'

labels = [
    {'name' : 'Hello', 'id':1},
    {'name' : 'Yes', 'id':2},
    {'name' : 'No', 'id':3},
    {'name' : 'Thank You', 'id':4},
    {'name' : 'I Love You', 'id':5}
]

with open(ANNOTATION_PATH + '\label_map.pbtxt', 'w') as f:
    for label in labels:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')

#print('python {' + SCRIPTS_PATH + '/generate_tfrecord.py} -x {' + IMAGE_PATH + '/train} -l {' + ANNOTATION_PATH + '/label_map.pbtxt} -o {' + ANNOTATION_PATH + '/train.record}')       
os.system('python3 ' + SCRIPTS_PATH + '/generate_tfrecord.py -x ' + IMAGE_PATH + '/train -l ' + ANNOTATION_PATH + '/label_map.pbtxt -o ' + ANNOTATION_PATH + '/train.record')


#!python {SCRIPTS_PATH + '/generate_tfrecord.py'} -x{IMAGE_PATH + '/test'} -l {ANNOTATION_PATH + '/label_map.pbtxt'} -o {ANNOTATION_PATH + '/test.record