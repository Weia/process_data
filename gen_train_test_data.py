import  os

import numpy as np
def _open_label_file(label_file):

    with open(label_file) as f:
        contents=f.readlines()

    return contents
def _parse_a_line(line):

    list_line=line.split(' ')
    imgName=list_line[0]

    float_labels=np.asarray([float(x) for x in list_line[1:-1]]).reshape(-1,2)
    #valid_label=float_labels[position,:]#有效label
    #print(imgName,valid_label)
    return imgName,float_labels

def gen_part_txt(full_label,img_dir,part_label):
    names=os.listdir(img_dir)
    full_contents=_open_label_file(full_label)
    part_content=[]
    for line in full_contents:
        img_name,_=_parse_a_line(line)
        if img_name in names:
            part_content.append(line)
    print(len(names),len(part_content))
    with open(part_label,'w') as f_part:
        f_part.writelines(part_content)
    pass



full_label='/media/weic/新加卷/标注文件/front/all_zero_label_no_re.txt'
img_dir='/media/weic/新加卷/数据集/数据集/学生照片/front-enhance'
part_label='./result/front_enhance.txt'

gen_part_txt(full_label,img_dir,part_label)