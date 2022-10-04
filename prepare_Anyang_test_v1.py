import os
from shutil import copyfile
import pdb
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import argparse


parser = argparse.ArgumentParser(description='Test')                               
parser.add_argument('--h', action="store_true", help="low resolution filter for height under 150") 
opt = parser.parse_args()

dict_train_label = pd.read_excel('./Anyang_data/Attribute_labeling.xlsx', sheet_name=1,engine='openpyxl')
list_train_label = list(dict_train_label.items())

ID_array = np.array(list_train_label[0][1])
Tube_array = np.array(list_train_label[1][1])
gender_array = np.array(list_train_label[3][1])
age_array = np.array(list_train_label[4][1])
tall_array = np.array(list_train_label[5][1])
hair_type_array = np.array(list_train_label[6][1])
hair_color_array = np.array(list_train_label[7][1])
top_type_array = np.array(list_train_label[8][1])
top_color_array = np.array(list_train_label[9][1])
bottom_type_array = np.array(list_train_label[10][1])
bottom_color_array = np.array(list_train_label[11][1])
item1_array =  np.array(list_train_label[12][1])
item2_array =  np.array(list_train_label[13][1])
item3_array =  np.array(list_train_label[14][1])
#item4_array =  np.array(list_train_label[15][1])

gender_list = ['female', 'male']
age_list = [0, 1, 2, 3, 4, 5]
tall_list = [11, 12, 13, 14, 15, 16, 17, 18]
hair_type_list = ['long', 'normal', 'permed', 'shaved', 'short', 'sporting', 'straight', 'tied_hair']
hair_color_list = ['black', 'brown', 'yellow']
top_type_list = ['long_sleeve', 'short_sleeve']
top_color_list = ['beige', 'black', 'blue', 'blue_green', 'brown', 'burgundy', 'dark_gray', 'gray',
        'green', 'khaki', 'light_brown', 'navy', 'navy_blue', 'navy_green', 'orange', 'pink',
        'purple', 'red', 'red_brown', 'sky_blue', 'white', 'yellow', 'yellow_green']
bottom_type_list = ['dress', 'long_pants', 'long_skirt', 'short_pants', 'short_skirt']
bottom_color_list = ['beige', 'black', 'blue', 'brown', 'burgundy', 'dark_gray', 'gray', 'navy',
        'navy_blue', 'navy_green', 'pink', 'purple', 'sky_blue', 'white']
item_list = ['backpack', 'bag', 'bicycle', 'carrier', 'cellphone', 'electric_wheel', 'etc',
        'glasses', 'hat', 'mask', 'motorcycle', 'plastic_cup', 'portable_fan', 'stroller',
        'sunglasses', 'umbrella', 'water_bottle']

# You need to change this line to your dataset download path
download_path = '/home/jicheol/Anyang_ar/Anyang_data/test'

if not os.path.isdir(download_path):
    print('please change the download_path')

# You need to change this line to your dataset save path
save_path = '/home/jicheol/Anyang_ar/Anyang_data' + '/pytorch'
if not os.path.isdir(save_path):
    os.mkdir(save_path)

test_path = download_path + '/'
if opt.h:
    test_save_path = save_path + '/test_high'
else:
    test_save_path = save_path + '/test'

if not os.path.isdir(test_save_path):
    os.mkdir(test_save_path)

none_tube = 0
filtered_img_count = 0

for root, dirs, files in tqdm(os.walk(test_path, topdown=True)):
    for name in files:
        if not name[-3:]=='jpg':
            continue
        src_path = root + '/' + name
        if opt.h:                                                               
            img = Image.open(src_path)                                          
            if img.size[1] < 150:                                               
                filtered_img_count+=1                                           
                continue

        ID = int(name[:6])
        Tube = int(name.split('T')[1][:4])
        
        index_list = np.where(ID_array==ID)[0]
        tube_index = np.where(Tube_array[index_list]==Tube)[0]
        if len(tube_index) != 1:
            if len(tube_index) == 0:
                none_tube +=1
                continue
            else:
                if Tube == 643:
                    tube_index = [tube_index[0]]
                else:
                    print("index_error")
                    pdb.set_trace()
        
        index = index_list[tube_index][0] 

        gender = gender_array[index]
        age = age_array[index]
        age = 5 if age >= 50 else age//10 
        tall = tall_array[index]
        tall= 11 if tall <120 else tall//10
        hair_type = hair_type_array[index]
        hair_color = hair_color_array[index]
        top_type = top_type_array[index]
        if not top_type in top_type_list:
            none_tube += 1
            continue
        top_color = top_color_array[index]
        bottom_type = bottom_type_array[index]
        bottom_color = bottom_color_array[index]
        if not bottom_color in bottom_color_list:
            none_tube += 1
            continue
        item1 = item1_array[index]
        item2 = item2_array[index]
        item3 = item3_array[index]
        #item4 = item4_array[index]

        gender_label = gender_list.index(gender)
        if hair_type == "long" or hair_type == "tied_hair":
            hair_type_label = 0
        elif hair_type == "normal":
            hair_type_label = 1
        else:
            hair_type_label = 2
        hair_color_label = hair_color_list.index(hair_color)
        top_type_label = top_type_list.index(top_type)
        if top_color == "beige" or top_color == "brown" or top_color == "light_brown":
            top_color_label = 0
        elif top_color == "black":
            top_color_label = 1
        elif top_color == "blue" or top_color == "blue_green" or top_color == "navy" or top_color == "navy_blue" or top_color == "sky_blue":
            top_color_label = 2
        elif top_color == "red" or top_color == "burgundy" or top_color == "red_brown":
            top_color_label = 3
        elif top_color == "gray" or top_color == "dark_gray":
            top_color_label = 4
        elif top_color == "green" or top_color == "navy_green" or top_color == "khaki":
            top_color_label = 5
        elif top_color == "orange" or top_color == "yellow" or top_color == "yellow_green":
            top_color_label = 6
        elif top_color == "pink":
            top_color_label = 7
        elif top_color == "purple":
            top_color_label = 8
        elif top_color == "white":
            top_color_label = 9
        if bottom_type == "dress" or bottom_type == "long_pants" or bottom_type == "long_skirt":
            bottom_type_label = 0
        else:
            bottom_type_label = 1
        if bottom_color == "beige" or bottom_color == "brown":
            bottom_color_label = 0
        elif bottom_color == "black":
            bottom_color_label = 1
        elif bottom_color == "blue" or bottom_color == "navy_blue" or bottom_color == "navy_green":
            bottom_color_label = 2
        elif bottom_color == "red" or bottom_color == "burgundy":
            bottom_color_label = 3
        elif bottom_color == "gray" or bottom_color == "dark_gray":
            bottom_color_label = 4
        elif bottom_color == "pink":
            bottom_color_label = 5
        elif bottom_color == "purple":
            bottom_color_label = 6
        elif bottom_color == "white":
            bottom_color_label = 7

        label_name = [gender_label, hair_type_label, hair_color_label,
                top_type_label, top_color_label, bottom_type_label, bottom_color_label]

        label_name = '_'.join(map(str,label_name))


        dst_path = test_save_path + '/' + label_name
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + label_name+"_"+name)

print(none_tube)
print(filtered_img_count)
