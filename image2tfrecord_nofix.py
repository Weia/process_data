import os
from PIL import Image
import tensorflow as tf
import numpy as np
import process_img
# WIDTH=256
# HEIGHT=256

def _read_images(img_list):
    """读取txt文件的images信息"""
    print(1)
    with open(img_list) as f:
        images=f.readlines()
        f.close()
    return images
def _parse_image(image):
    """解析一行images信息"""
    image = image.split(' ')
    imgName = image[0]#图像名
    label = image[1:-1]
    label = np.asarray([float(x) for x in label])
    return imgName,label

def img2TfRecord(img,label):
    """
    将image和label转换成tfrecord的example
    :param img:
    :param label: float list
    :return:
    """
    # 转换成二进制
    width,height=img.size
    # print(width,height)
    f_label = (label.reshape(1, -1)).tolist()[0]
    img_raw = img.tobytes()
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'w':tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
                'h': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
                'label': tf.train.Feature(float_list=tf.train.FloatList(value=f_label)),
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
    s_example=example.SerializeToString()
    return s_example

def createTfRecords(img_dir,filename,img_list):
    """
        根据保存图像名和label的txt文件生成tfrecords文件

        :param img_dir: 存储图像路径
        :param filename: 要生成tfrecords文件名
        :param img_list: 保存图像的list
    """
    writer=tf.python_io.TFRecordWriter(filename)
    images=_read_images(img_list)
    #imgNum=len(images)
    i=1
    for image in images:
        print(i)
        #解析image获得图像名和label
        imgName,label=_parse_image(image)

        or_image=Image.open(os.path.join(img_dir,imgName))
        width,height=or_image.size
        #从图像中裁剪出人
        #img,label=process_img.crop_data(or_image,label)
        # label=label.reshape(-1,2)
        #print(real_image)
        #将图像转换成256*256
        # re_image,re_label=process_img.resize_image(or_image,label)


        #print(re_image.size)
        #real_image=re_image.resize((WIDTH,HEIGHT),Image.ANTIALIAS)

        #将image和label转换成example
        re_label=label.reshape(-1,2)
        w=[1 if x>width else 0 for x in re_label[:,0]]
        h=[1 if x>height else 0 for x in re_label[:,1]]
        if 1 in w or 1 in h:
            print('label error')
            continue


        example=img2TfRecord(or_image,label)
        writer.write(example)
        i+=1




    writer.close()


createTfRecords(r'/media/weic/新加卷/数据集/数据集/学生照片/front-valid',
                '/media/weic/新加卷/数据集/数据集/学生照片/tfrecords/front/final_valid.tfrecords',
                '/home/weic/project/deal_with_lable/project/front/final_valid_label.txt')




