import tensorflow as tf
import numpy as np



def adjust_bright(img):
    #0,0.5
    print(1)
    return tf.image.random_brightness(img,32./255.)

def adjust_const(img):
    #0.5-2
    # return tf.image.adjust_contrast(img,2)
    print(2)
    return tf.image.random_contrast(img,0.5,1.5)

def adjust_sat(img):
    #0-2
    print(3)
    return tf.image.random_saturation(img,0.5,1.5)




def gen_full_boundingBox(label,width,height):

    re_width=tf.cast(width,tf.int32)
    re_height=tf.cast(height    ,tf.int32)
    re_label=tf.reshape(label,(-1,2))
    l_min=tf.reduce_min(re_label,axis=0)
    l_max=tf.reduce_max(re_label,axis=0)
    left_margin=tf.cast(tf.floor(l_min[0]),tf.int32)
    top_margin=tf.cast(tf.floor(l_min[1]),tf.int32)
    right_margin=tf.cast(tf.floor(l_max[0]),tf.int32)
    bottom_margin=tf.cast(tf.floor(l_max[1]),tf.int32)

    left0=tf.random_uniform([1],0,left_margin+1,dtype=tf.int32)
    left=tf.random_uniform([1],0,left0[0]+1,dtype=tf.int32)
    #left=tf.random_uniform([1],0,left1[0]+1,dtype=tf.int32)

    top0=tf.random_uniform([1],0,top_margin+1,dtype=tf.int32)
    top=tf.random_uniform([1],0,top0[0]+1,dtype=tf.int32)
    # top=tf.random_uniform([1],0,top1[0]+1,dtype=tf.int32)

    right0=tf.random_uniform([1],right_margin,re_width,dtype=tf.int32)
    right=tf.random_uniform([1],right0[0],re_width,dtype=tf.int32)
    # right=tf.random_uniform([1],right1[0],re_width,dtype=tf.int32)


    bottom0=tf.random_uniform([1],bottom_margin,re_height,dtype=tf.int32)
    bottom=tf.random_uniform([1],bottom0[0],re_height,dtype=tf.int32)
    # bottom=tf.random_uniform([1],bottom1[0],re_height,dtype=tf.int32)
    new_width=right-left
    new_height=bottom-top

    return (top[0],left[0],new_height[0],new_width[0])

def relabel_ac_bbox(label,bbox):
    re_label=tf.reshape(label,(-1,2))


    top=tf.cast(bbox[0],tf.float32)
    left=tf.cast(bbox[1],tf.float32)
    x=tf.reshape(re_label[:,0]-left,(-1,1))
    y=tf.reshape(re_label[:,1]-top,(-1,1))
    result=tf.concat([x,y],axis=1)
    return result




def random_crop_img(img,label,width,height):
    bbox=gen_full_boundingBox(label,width,height)
    crop_img=tf.image.crop_to_bounding_box(img,bbox[0],bbox[1],bbox[2],bbox[3])

    re_label=relabel_ac_bbox(label,bbox)
    return crop_img,re_label,bbox[2],bbox[3]

    pass

def adjust_image(color_ordering,image,fast=False):
    if fast:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    else:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)

      elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)

      elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)

      elif color_ordering == 3:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    return image
