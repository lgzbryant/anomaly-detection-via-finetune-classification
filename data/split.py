#_*_coding:utf-8_*_

#author: lgz

#date: 19-6-10



import os
import shutil


def objFileName():
    obj_name_list = { }

    for i in range(10):
        # print(i)
        if i>0:
            obj_name_list[str(i)] = []

            for filename in os.listdir(r"./test_cifar10"):
                if int(filename.split('_')[0]) == i and len(obj_name_list[str(i)])<=112:
                    obj_name_list[str(i)].append(filename)

                # print(filename)

    return obj_name_list


def copy_img():
    local_img_name = r"./test_cifar10"
    i = 1
    path = r'./val/1to9'
    if os.path.exists(path) == False:
        os.mkdir(path)
    for list in objFileName():
        print(list)
        # path = r'./train/'+str(i)
        # if os.path.exists(path) == False:
        #     os.mkdir(path)

        for file in objFileName()[str(i)]:
            print(local_img_name + '/' + file)
            shutil.copy(local_img_name + '/' + file, path + '/' + file)
        i += 1


if __name__ == '__main__':
    copy_img()





#####################
# def objFileName():
#     obj_name_list = { }
#
#     for i in range(10):
#         # print(i)
#         obj_name_list[str(i)] = [ ]
#
#         for filename in os.listdir(r"./train_cifar10"):
#             if int(filename.split('_')[0]) == i :
#                 obj_name_list[str(i)].append(filename)
#
#                 # print(filename)
#
#     return obj_name_list
#
#
# def copy_img():
#     local_img_name = r"./train_cifar10"
#     i = 0
#     for list in objFileName():
#         print(list)
#         path = r'./train/'+str(i)
#         if os.path.exists(path) == False:
#             os.mkdir(path)
#
#         for file in objFileName()[str(i)]:
#             print(local_img_name + '/' + file)
#             shutil.copy(local_img_name + '/' + file, path + '/' + file)
#         i += 1
#
#
# if __name__ == '__main__':
#     copy_img()

    # print(objFileName()['2'])