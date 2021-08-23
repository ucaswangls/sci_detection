from torch.utils.data import Dataset
import cv2
import os 
import os.path as osp
import numpy as np 
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
class VIDData(Dataset):
    def __init__(self,vid_dir = "F:/datasetes/VID2015/imagenet2015/ILSVRC"):
        self.classes = ['__background__',  # always index 0
                    'airplane', 'antelope', 'bear', 'bicycle',
                    'bird', 'bus', 'car', 'cattle',
                    'dog', 'domestic_cat', 'elephant', 'fox',
                    'giant_panda', 'hamster', 'horse', 'lion',
                    'lizard', 'monkey', 'motorcycle', 'rabbit',
                    'red_panda', 'sheep', 'snake', 'squirrel',
                    'tiger', 'train', 'turtle', 'watercraft',
                    'whale', 'zebra']
        self.classes_map = ['__background__',  # always index 0
                    'n02691156', 'n02419796', 'n02131653', 'n02834778',
                    'n01503061', 'n02924116', 'n02958343', 'n02402425',
                    'n02084071', 'n02121808', 'n02503517', 'n02118333',
                    'n02510455', 'n02342885', 'n02374451', 'n02129165',
                    'n01674464', 'n02484322', 'n03790512', 'n02324045',
                    'n02509815', 'n02411705', 'n01726692', 'n02355227',
                    'n02129604', 'n04468005', 'n01662784', 'n04530566',
                    'n02062744', 'n02391049']
        video_all_dir = os.path.join(vid_dir,"Data/VID/train")
        label_all_dir = os.path.join(vid_dir,"Annotations/VID/train")
        train_file_dir = os.path.join(vid_dir,"ImageSets/VID")
        self.video_dir_list = []
        self.label_dir_list = []
        for train_file_name in os.listdir(train_file_dir):
            train_file_name = os.path.join(train_file_dir,train_file_name)
            if train_file_name[-11:-6] != "train":
                continue
            with open(train_file_name,"r") as fp:
                lines = fp.readlines()
            for line in lines:
                _name,_ = line.strip().split()
                video_dir = os.path.join(video_all_dir,_name)
                if len(os.listdir(video_dir))<8:
                    print("image num: {} < 8.",video_dir)
                    continue
                self.video_dir_list.append(video_dir)
                label_dir = os.path.join(label_all_dir,_name)
                self.label_dir_list.append(label_dir)
        
    def __getitem__(self, index):
        image_dir = self.video_dir_list[index]
        label_dir = self.label_dir_list[index]
        image_name_list = os.listdir(image_dir)
        image_name_list.sort(key= lambda x:int(x[:-5]))
        label_name_list = os.listdir(label_dir)
        label_name_list.sort(key= lambda x:int(x[:-4]))
        for ii,image_name in enumerate(image_name_list):
            image = cv2.imread(osp.join(image_dir,image_name))

            tree = ET.parse(os.path.join(label_dir,label_name_list[ii])).getroot()
            anns,labels,trackids = self._preprocess_annotation(tree)
            if len(anns)==0:
                print(label_name_list[ii])
            if len(anns)==1:
                continue
            print("ii:",ii)
            if ii>3:
                break
            for i,bbox in enumerate(anns):
                
                x1,y1,x2,y2 = bbox
                label = labels[i]
                trackid = trackids[i]
                cv2.putText(image,str(label)+"_"+str(trackid),(x1,y1),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255))
                cv2.rectangle(image,(x1,y1),(x2,y2),(255,0,0))
            plt.imshow(image)

            plt.show()

    def _preprocess_annotation(self,target):
        classes_to_ind = dict(zip(self.classes_map, range(len(self.classes_map))))
        size = target.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))

        objs = target.findall("object")
        anns = []
        labels = []
        trackids = []
        for obj in objs:
            if not obj.find("name").text in classes_to_ind:
                continue
            trackid = int(obj.find("trackid").text)
            bbox =obj.find("bndbox")
            box = [
                np.maximum(int(bbox.find("xmin").text), 0),
                np.maximum(int(bbox.find("ymin").text), 0),
                np.minimum(int(bbox.find("xmax").text), im_info[1] - 1),
                np.minimum(int(bbox.find("ymax").text), im_info[0] - 1)
            ]
            anns.append(box)
            labels.append(classes_to_ind[obj.find("name").text.lower().strip()])
            trackids.append(trackid)
        return anns,labels,trackids
    def __len__(self,):
        return len(self.video_dir_list)
            

if __name__=="__main__":
    datasets = VIDData(vid_dir="/media/wangls/new_disk/datasetes/VID2015/imagenet2015/ILSVRC")
    for data in datasets:
        print("hello")
