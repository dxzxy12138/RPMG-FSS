import os
import glob

if __name__ == '__main__':
    # /data/zxy/meta-setr/PFENet/dataset/coco_data/train2014/COCO_train2014_000000167126.jpg
    # /data/zxy/meta-setr/PFENet/dataset/coco_data/train/COCO_train2014_000000167126.png
    
    # /data1/zhicheng/PFENet/coco_data/train2014/COCO_train2014_000000494112.jpg 
    # /data1/zhicheng/PFENet/coco_data/train/COCO_train2014_000000494112.png
    coco_train_file_path = '/data/zxy/meta-setr/PFENet/lists/coco_2/train_data_list.txt'
    coco_test_file_path = '/data/zxy/meta-setr/PFENet/lists/coco_2/val_data_list.txt'
    
    new_coco_train_file_path = '/data/zxy/meta-setr/PFENet/lists/coco_2/new_train_data_list.txt'
    mew_coco_test_file_path = '/data/zxy/meta-setr/PFENet/lists/coco_2/new_val_data_list.txt'
    
    coco_train_file_list, coco_test_file_list = list(), list()
    with open(coco_train_file_path, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')  #去掉列表中每一个元素的换行符
            # print(line)
            line = line.split(' ')
            # print(line)
            # line[0] = '/data/zxy/meta-setr/PFENet/dataset' + line[0][22:]
            # line[1] = '/data/zxy/meta-setr/PFENet/dataset' + line[1][22:]
            line[0] = line[0][33:]
            line[1] = line[1][33:]
            # print(line[0])
            # line[0].replace('PFENet', '/PFENet/dataset/')
            # line[1].replace('PFENet', '/PFENet/dataset/')
            # print(line)
            line = ' '.join(line)
            coco_train_file_list.append(line)
    
    with open(coco_test_file_path, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')  #去掉列表中每一个元素的换行符
            line = line.split(' ')
            line[0] = line[0][33:]
            line[1] = line[1][33:]
            line = ' '.join(line)
            coco_test_file_list.append(line)
    
    sstr = '\n'
    with open(new_coco_train_file_path, "w") as f:
        f.write(sstr.join(coco_train_file_list))
    
    sstr = '\n'
    with open(mew_coco_test_file_path, "w") as f:
        f.write(sstr.join(coco_test_file_list))