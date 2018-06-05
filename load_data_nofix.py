import tensorflow as tf
import data_enhance
import numpy as np
"""加载一个batchsize的image"""
WIDTH=256
HEIGHT=256
HM_HEIGHT=64
HM_WIDTH=64
def _read_single_sample(samples_dir,nPoints):

    filename_quene=tf.train.string_input_producer([samples_dir])
    reader=tf.TFRecordReader()
    _,serialize_example=reader.read(filename_quene)
    features=tf.parse_single_example(
        serialize_example,
        features={
                    'w': tf.FixedLenFeature([], tf.int64),
                    'h': tf.FixedLenFeature([], tf.int64),
                    'label':tf.FixedLenFeature([nPoints*2],tf.float32),
                    'image':tf.FixedLenFeature([],tf.string)
        }
    )

    width = tf.cast(features['w'], tf.int32)
    height = tf.cast(features['h'], tf.int32)
    image = tf.decode_raw(features['image'], tf.uint8)

    image = tf.reshape(image, [height,width, 3])#！reshape 先列后行
    label = tf.cast(features['label'], tf.float32)


    return image,label,width,height

def resize_img_label(image,label,width,height):
    new_img=tf.image.resize_images(image,[WIDTH,HEIGHT],method=1)
    x=tf.reshape(label[:,0]*256./tf.cast(width,tf.float32),(-1,1))
    y=tf.reshape(label[:,1]*256./tf.cast(height,tf.float32),(-1,1))
    re_label=tf.concat([x,y],axis=1)
    return new_img,re_label

def batch_samples(batch_size,filename,nPoints,shuffle=False):
    """
    filename:tfrecord文件名
    """

    image,label,width,height=_read_single_sample(filename,nPoints)
    # print(image.shape)
    # label=tf.reshape(label,[-1,2])
    image,new_label,new_height,new_width=data_enhance.random_crop_img(image,label,width,height)
    image,label=resize_img_label(image,new_label,new_width,new_height)
    # new_image=tf.image.resize_images(image,[256,256],method=1)
    # new_image=tf.reshape(new_image,[256,256,3])

    # new_image=data_enhance.adjust_image(2,new_image)
    #label = gene_hm.resize_label(label)#将label放缩到64*64
    #label=gene_hm.tf_generate_hm(HM_HEIGHT, HM_WIDTH ,label, 64)
    # if shuffle:
    #     image_batch, label_batch = tf.train.shuffle_batch([new_image, label], batch_size, min_after_dequeue=10,num_threads=2,capacity=2000)
    # else:
    #     image_batch,label_batch=tf.train.batch([new_image,label],batch_size, num_threads=2)

    return image,label



# # # # """测试加载图像"""
import matplotlib.pyplot as plt
#import load_batch_data
from PIL import Image
#import numpy as np
#from pyheatmap import HeatMap


with tf.Session() as sess: #开始一个会话
    init_op = tf.global_variables_initializer()


    image_batch,label_batch=batch_samples(2,r'/media/weic/新加卷/数据集/数据集/学生照片/tfrecords/front/final_valid.tfrecords',16,False)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    sess.run(init_op)
    for j in range(29):
        example, l = sess.run([image_batch, label_batch])  # 在会话中取出image和label
        img=Image.fromarray(example)

        # width, height = img.size
        # img = img.resize((256, 256))
        label = l.reshape(-1, 2)
        # label[:, 0] = label[:, 0] * 256 / width
        # label[:, 1] = label[:, 1] * 256 / height
        plt.imshow(img, cmap='Greys_r')
        plt.plot(label[:, 0], label[:, 1], 'r+')


        plt.show()
    coord.request_stop()
    coord.join(threads)
