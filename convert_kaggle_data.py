import sys
import os
import shutil
import pandas as pd


if len(sys.argv) != 2:
    print("""USAGE:
    python convert_kagle_data.py [dir]
    - dir - path to directory, where Kaggle data is located""")
    exit(-1)

input_dir = sys.argv[1]
output_dir = './cifar-dataset/kaggle/'
train_dir = f'{output_dir}train/'
test_dir = f'{output_dir}test/'
test_class_dir = f'{test_dir}unknown/'

# create required directories if they do not exist
for dir in [output_dir, train_dir, test_dir, test_class_dir]:
    if not os.path.isdir(dir):
        os.mkdir(dir)

# create directories for each label
labels_df = pd.read_csv(f'{input_dir}trainLabels.csv')
for lab in labels_df['label'].unique():
    lab_dir = f'{train_dir}{lab}/'
    if not os.path.isdir(lab_dir):
        os.mkdir(lab_dir)

# copy each train image to refering directory
for row in labels_df.values:
    id, label = row
    old_path = f'{input_dir}train/{id}.png'
    new_path = f'{train_dir}{label}/{str(id).zfill(7)}.png'
    shutil.copy(old_path, new_path)
    print(f'Image copied from {old_path} to {new_path}')


# copy all test images to single label directory
for file in os.listdir(f'{input_dir}test/'):
    id = os.path.splitext(file)[0]
    old_path = f'{input_dir}test/{file}'
    new_path = f'{test_class_dir}{id.zfill(7)}.png'
    if os.path.isfile(old_path):
        shutil.copy(old_path, new_path)
        print(f'Image copied from {old_path} to {new_path}')
