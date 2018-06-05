生成tfrecords，数据增强，



image2tfrecords.py 将image写入tfrecords文件中，制作tensorflow的数据集

process_img.py 对image进行预处理，包括，裁剪，resize img和label，主要image2tfrecords.py 的辅助函数

load_batch_data.py 测试产生的tfrecords文件，方法，加载一个batch



left-arm 有些问题，变形
image2tfrecord_nofix.py 创建tfrecords文件中image不是固定大小的图片

load_data_nofix.py 从不是存储不是固定大小图片的tfrecoeds读取图片
