#coding=utf-8
import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
import codecs
import os
import argparse
import numpy as np
from scipy.misc import imread


def load_bboxes(label_path, width, height, label_map=None,absCoord=False):
    objs = []
    if absCoord:
        width, height = 1.0, 1.0
    with codecs.open(label_path, 'r', 'utf-8') as f:
        for line in f:
            obj = {}
            label, c_x, c_y, w, h = line.strip('\n').split()
            label = int(label)
            if label_map:
                label = label_map[label]
            obj['label'] = label
            obj['bbox'] = list(map(float, [c_x, c_y, w, h]))
            obj['bbox'][0] *= width
            obj['bbox'][2] *= width
            obj['bbox'][1] *= height
            obj['bbox'][3] *= height
            objs.append(obj)
    return objs


def load_label_map(label_file):
    ret = []
    with codecs.open(label_file, 'r', 'utf-8') as f:
        for line in f:
            ret.append(line.strip('\n'))
    label_str_map = dict([(i, label) for i,label in enumerate(ret)])
    str_label_map = dict([(label, i) for i,label in enumerate(ret)])
    return label_str_map, str_label_map


def get_image_name(img_path):
    return os.path.basename(img_path)



def draw_detection(img_path, label_path, savedir, num2label_map=None, label2num_map=None, absCoord=False):
    img = imread(img_path)
    h, w, c =  img.shape
    img_name =  get_image_name(img_path)
    bboxes = load_bboxes(label_path, w, h, num2label_map,absCoord)
    plt.clf()
    plt.imshow(img)
    plt.axis('off');
    ax = plt.gca()

    num_classes = len(num2label_map) if num2label_map else 15
    colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()
    
    for obj in bboxes:
        
        label = obj['label']
        color_num = label2num_map[label] if label2num_map else int(label)
        color = colors[color_num % num_classes]
        
        bbox = obj['bbox']
        coords = (bbox[0] - bbox[2]/2., bbox[1]-bbox[3]/2.), bbox[2], bbox[3]
        ax.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=1))
        if 'score' in obj:
            score = obj['score']
            display_text = '%s: %.2f' % (label, score)
        else:
            display_text = str(label)
            ax.text(coords[0][0], coords[0][1], display_text, bbox={'facecolor':color, 'alpha':0.5})
    if savedir and bboxes:
        save_img = os.path.join(savedir, img_name)
        plt.savefig(save_img, bbox_inches="tight")
        print "saving image:%s" % save_img
#    plt.show()


def isImage(path):
    return path.lower().endswith(".jpg") or path.lower().endswith(".png") or path.lower().endswith(".jpeg")

def isText(path):
    return path.lower().endswith(".txt")

def create_if_not_exists(path_name, isdir=True):
    if os.path.exists(path_name):
        return
    os.mkdir(path_name) if isdir else os.mknod(path_name)

def get_label_path_by_image_path(image_path):
    txt_path = image_path.replace(".jpg",".txt")
    txt_path = txt_path.replace(".png",".txt")
    txt_path = txt_path.replace(".jpeg",".txt")
    txt_path = txt_path.replace("JPEGImages","labels")
    return txt_path

def get_image_path_by_label_path(label_path):
    img_path = label_path.replace(".txt",".jpg")
    img_path = img_path.replace("labels","JPEGImages")
    #img_path = img_path.replace("labels","images")
    return img_path


def draw_detections(img_path_or_dir, txt_path_or_dir, label_map, savedir, find_image_by_label=True, absCoord=False):
    image_paths, txt_paths = [], []
    if not os.path.exists(img_path_or_dir):
        raise Exception("Error:%s not exists"%img_path_or_dir)
    if os.path.isdir(img_path_or_dir):
        assert os.path.isdir(txt_path_or_dir), "label must be dir too"
        if find_image_by_label:
            for tx in os.listdir(txt_path_or_dir):
                if isText(tx):
                    tx_path = os.path.join(txt_path_or_dir,tx)
                    image_paths.append(get_image_path_by_label_path(tx_path))
                    txt_paths.append(tx_path)
        else:
            for im in os.listdir(img_path_or_dir):
                if isImage(im):
                    im_path = os.path.join(img_path_or_dir,im)
                    image_paths.append(im_path)
                    txt_paths.append(get_label_path_by_image_path(im_path))

    if os.path.isfile(img_path_or_dir):
        assert os.path.isfile(txt_path_or_dir), "label must be file too"
        image_paths = [img_path_or_dir]
        txt_paths = [txt_path_or_dir]

    if not os.path.exists(savedir):
        create_if_not_exists(savedir)


    num2label_map, label2num_map = None, None
    if label_map:
        num2label_map, label2num_map = load_label_map(label_map)
    for img, txt in zip(image_paths, txt_paths):
        draw_detection(img, txt, savedir, num2label_map, label2num_map, absCoord)



if __name__=="__main__":
    
    """
    python python/plot_detections.py /Users/guoqingpei/Desktop/sharedData/JPEGImages /Users/guoqingpei/Desktop/sharedData/labels -label_map /Users/guoqingpei/Desktop/darkness/data/sunshang.names -save_dir /Users/guoqingpei/Desktop/sharedData/test
    """
    parser = argparse.ArgumentParser() #获取参数过滤器
    parser.add_argument("image_file_or_dir",type=str,help="image_file_or_dir does not exists")
    parser.add_argument("label_file_or_dir",type=str,help="image_file_or_dir does not exists")
    parser.add_argument("-label_map","--label_map",type=str,default=None,help="label conversion")
    parser.add_argument("-save_dir","--save_dir",type=str,default=None,help="labeled images save dir")
    parser.add_argument("-find_image_by_label","--find_image_by_label",type=bool,default=True,help="search images by label and vice versa")
    parser.add_argument("-absCoord","--absCoord",type=bool,default=False,help="absolute coords in label file")
    
    
    args = parser.parse_args() #实际从命令行中提取参数的过程

    draw_detections(args.image_file_or_dir,                                 args.label_file_or_dir,
                    args.label_map, args.save_dir, args.find_image_by_label,args.absCoord)


