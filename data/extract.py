#_*_coding:utf-8_*_

#author: lgz

#date: 19-6-10


import os
import shutil

def objFileName():
    obj_name_list = []
    for filename in os.listdir(r"./Concrete Crack Images for Classification/Positive_1"):
        if int(filename.split('.')[0].split('_')[0])<= 2000 :
            obj_name_list.append(filename)
            print(filename)
    return obj_name_list


def copy_img():
    local_img_name = r"./Concrete Crack Images for Classification/Positive_1"
    # 指定要复制的图片路径
    path = r'./test_crack/train/1'
    # 指定存放图片的目录
    for i in objFileName():
        new_obj_name = i
        shutil.copy(local_img_name + '/' + new_obj_name, path + '/' + new_obj_name)


if __name__ == '__main__':
    copy_img()