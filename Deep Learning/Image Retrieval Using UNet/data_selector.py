import os
import random
import shutil


TOTAL_DATA = 1000
ORIGIN_PATH = './CIFAR-10-images/'
DST_PATH = './generated_images/'

num_train, num_test = int(0.9 * TOTAL_DATA), int(0.1 * TOTAL_DATA)
train_path, test_path = ORIGIN_PATH + 'train/', ORIGIN_PATH + 'test/'
num_classes = len(os.listdir(train_path))

shutil.rmtree(DST_PATH)
os.makedirs(DST_PATH, exist_ok=True)
os.makedirs(DST_PATH + 'train/', exist_ok=True)
os.makedirs(DST_PATH + 'test/', exist_ok=True)

# generation training data
instance_per_class = num_train // num_classes
idx = 0
for class_name in os.listdir(train_path):
    path = train_path + class_name
    images = random.sample(os.listdir(path), instance_per_class)
    for image in images:
        org_path = path + '/' + image
        dst_path = DST_PATH + 'train/' + str(idx) + '.jpg'
        shutil.copy(org_path, dst_path)
        idx += 1

# generation test data
instance_per_class = num_test // num_classes
idx = 0
for class_name in os.listdir(test_path):
    path = test_path + class_name
    images = random.sample(os.listdir(path), instance_per_class)
    for image in images:
        org_path = path + '/' + image
        dst_path = DST_PATH + 'test/' + str(idx) + '.jpg'
        shutil.copy(org_path, dst_path)
        idx += 1
