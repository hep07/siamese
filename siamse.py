import os
import pickle

file_path_header = "../../Columbia_2017/scrapy-nap/nap_store1"

# read proprocessed product dictionary
def load_obj(file_path):
    with open(file_path + '.pkl', 'rb') as f:
        return pickle.load(f)

processed_product_dict = load_obj(file_path_header + "/processed_dict")


product_code = processed_product_dict.keys()
# provide randome seed to shuffle function
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split




shuffle_seed = random.randrange(0,1)
shuffle(product_code) =

# form dict image file name, product code while removing images to be taken out from training
img_dict = dict()
for key, value in processed_product_dict.items():
    filename_arr = [img['path'].split('/')[1].split('.')[0] for img in value['images'] if img['url'].split('_')[1] not in ['e3','ou']]
    url_arr = [img['url'].split('/')[1].split('.')[0] for img in value['images']]
    for filename in filename_arr:
        if filename not in img_dict:
            img_dict[filename] = value['product_code']


list(img_dict.keys())
img_dict.keys().
# train test split

X_train_valid,X_test = train_test_split(list(img_dict.keys()),test_size = 0.15 , random_state=2018)

X_train, X_valid = train_test_split(X_train_valid,test_size = 0.2 , random_state=2018)


len(X_train)
len(X_test)
len(X_test)

from shutil import copyfile

os.makedirs('./180112/train/')
os.makedirs('./180112/valid/')
os.makedirs('./180112/test/')

for img in X_train:
    src = file_path_header + "/full/" + img +'.jpg'
    dst_folder = './180112/train/'  + img_dict[img]
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    dst_img = dst_folder + '/' + img +'.jpg'
    copyfile(src, dst_img)

for img in X_valid:
    src = file_path_header + "/full/" + img +'.jpg'
    dst_folder = './180112/valid/'  + img_dict[img]
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    dst_img = dst_folder + '/' + img +'.jpg'
    copyfile(src, dst_img)

for img in X_test:
    src = file_path_header + "/full/" + img +'.jpg'
    dst_folder = './180112/test/'  + img_dict[img]
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    dst_img = dst_folder + '/' + img +'.jpg'
    copyfile(src, dst_img)

