import tensorflow as tf
import data_enhance
import numpy as np
"""加载一个batchsize的image"""
WIDTH=512
HEIGHT=512
HM_HEIGHT=64
HM_WIDTH=64
def _read_single_sample(samples_dir,nPoints):

    filename_quene=tf.train.string_input_producer([samples_dir])
    reader=tf.TFRecordReader()
    _,serialize_example=reader.read(filename_quene)
    features=tf.parse_single_example(
        serialize_example,
        features={
                    'label':tf.FixedLenFeature([nPoints*2],tf.float32),
                    'image':tf.FixedLenFeature([],tf.string)
        }
    )


    image = tf.decode_raw(features['image'], tf.uint8)
    image = tf.reshape(image, [HEIGHT,WIDTH, 3])#！reshape 先列后行
    label = tf.cast(features['label'], tf.float32)


    return image,label


def resize_img_label(image,label,width,height):
    new_img=tf.image.resize_images(image,[256,256],method=1)
    x=tf.reshape(label[:,0]*256./tf.cast(width,tf.float32),(-1,1))
    y=tf.reshape(label[:,1]*256./tf.cast(height,tf.float32),(-1,1))
    re_label=tf.concat([x,y],axis=1)
    return new_img,re_label

def batch_samples(batch_size,filename,nPoints,shuffle=False):
    """
    filename:tfrecord文件名
    """

    image,label=_read_single_sample(filename,nPoints)
    # print(image.shape)
    # label=tf.reshape(label,[-1,2])

    image,label,re_width,re_height=data_enhance.do_enhance(image,label,512,512)
    image,label=resize_img_label(image,label,re_width,re_height)
    #label = gene_hm.resize_label(label)#将label放缩到64*64
    #label=gene_hm.tf_generate_hm(HM_HEIGHT, HM_WIDTH ,label, 64)
    if shuffle:
        image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size, min_after_dequeue=10,num_threads=2,capacity=2000)
    else:
        image_batch,label_batch=tf.train.batch([image,label],batch_size, num_threads=2)

    return image_batch,label_batch



# # # # """测试加载图像"""
import matplotlib.pyplot as plt
#import load_batch_data
# from PIL import Image
#import numpy as np
#from pyheatmap import HeatMap


with tf.Session() as sess: #开始一个会话
    init_op = tf.global_variables_initializer()


    image_batch,label_batch=batch_samples(2,r'/media/weic/新加卷/数据集/数据集/学生照片/tfrecords/front/final_valid_512.tfrecords',16,False)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    sess.run(init_op)
    for j in range(29):
        example, l = sess.run([image_batch, label_batch])  # 在会话中取出image和label


        # print("1",l.shape)


        for i in range(2):
            #img=Image.fromarray(example[i], 'RGB')#这里Image是之前提到的
            #img.save('./testimg'+str(i)+'.jpg')#存下图片
            #print(l[i])

            #label = gene_hm.generate_hm(HM_HEIGHT, HM_WIDTH, l[i], 64)
            #print(label.shape)
            #label=np.sum(l[i],axis=0)
            #print(label.shape)
            #print(label)
            #x=l[i][:,0]
            #y=l[i][:,1]
            #print(img)
            # img = img.convert('L')
            # img=np.array(img)/255

            #print(example[i])
            #print(img)
            # plt.matshow(l[i], fignum=0)
            plt.imshow(example[i],cmap='Greys_r')
            plt.plot(l[i][:,0],l[i][:,1],'r*')
            # print(l[i])

            #print(len(label.tolist()))
            #hm=HeatMap(label)
            #for i in range(16):

            plt.show()
#     #print(example, l)
    coord.request_stop()
    coord.join(threads)
