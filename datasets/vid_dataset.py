from torch.utils.data import Dataset
import torch
import cv2
import os 
import os.path as osp
import numpy as np 
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from utils.image import gaussian_radius, draw_umich_gaussian 
from utils.auguments import transforms
import math 

class TrainData(Dataset):
    def __init__(self,opt):
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
        data_dir= opt.train_data_dir
        image_all_dir = osp.join(data_dir,"Data/VID/train")
        image_path_list = os.listdir(image_all_dir)
        label_all_dir = osp.join(data_dir,"Annotations/VID/train")
        self.img_files = []
        self.label_files = []
        count = 0
        for image_path in image_path_list:
            label_path = image_path
            count += 1
            print("data count:",count)
            image_path = osp.join(data_dir,image_all_dir,image_path)
            for image_dir in os.listdir(image_path):
                label_dir = image_dir
                image_dir = osp.join(image_path,image_dir)
                temp_list = os.listdir(image_dir)
                temp_list.sort(key=lambda x:int(x.split(".")[0]))
                meas_list = []
                label_list = []
                for i,image_name in enumerate(temp_list):
                    meas_list.append(osp.join(image_dir,image_name))
                    label_list.append(osp.join(label_all_dir,label_path,label_dir,image_name.replace(".jpg",".xml").replace(".JPEG",".xml")))
                    if (i+1)%opt.ratio==0:
                        self.img_files.append(meas_list)
                        self.label_files.append(label_list)
                        meas_list = []
                        label_list = []
                    if i>32:
                        break
        self.mask = opt.mask
        self.ratio,self.resize_h,self.resize_w = self.mask.shape
        self.output_h,self.output_w = self.resize_h//opt.down_ratio,self.resize_w//opt.down_ratio
        self.opt = opt
        self.num_classes = opt.num_classes
        self.max_objs = opt.max_objs
        self.draw_gaussian = draw_umich_gaussian
        
    def __getitem__(self, index):
        gt = np.zeros([self.ratio, self.resize_h, self.resize_w],dtype=np.float32)
        meas = np.zeros([self.resize_h, self.resize_w],dtype=np.float32)
        box_dict = {}
        frames_bbox_list = []
        gt_images_list = []

        image = cv2.imread(self.img_files[index][0])
        im_h,im_w,_ = image.shape
        transform = transforms(im_h=im_h,im_w=im_w)
        for i,image_path in enumerate(self.img_files[index]):
            label_path = self.label_files[index][i]
            tree = ET.parse(label_path).getroot()
            anns,labels,trackids = self._preprocess_annotation(tree)
            image = cv2.imread(image_path)
            per_frame_bbox_list = []
            category_ids = []
            for id_index,bbox in enumerate(anns):
                id = trackids[id_index]
                # cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255))
                x1,y1,x2,y2 = bbox
                bbox = np.array([x1/im_w,y1/im_h,x2/im_w,y2/im_h])
                bbox = np.clip(bbox,a_min=0,a_max=1).tolist()
                x1,y1,x2,y2 = bbox
                if x1>=x2 or y1>=y2:
                    continue
                per_frame_bbox_list.append(bbox)
                category_ids.append(id)
            #transform
            transformed = transform(image=image, bboxes=per_frame_bbox_list,category_ids=category_ids)
            image = transformed["image"]
            per_frame_bbox_list = transformed["bboxes"]

            per_frame_id_list = transformed["category_ids"]
            for id_index,id in enumerate(per_frame_id_list):
                if id not in box_dict.keys():
                    box_dict[id] = [] 
                box_dict[id].append(per_frame_bbox_list[id_index])

            frames_bbox_list.append(per_frame_bbox_list)
            image = cv2.resize(image,(self.resize_w,self.resize_h))
            pic_t = cv2.cvtColor(image,cv2.COLOR_BGR2YCrCb)[:,:,0]
            gt_images_list.append(pic_t)
            pic_t = pic_t.astype(np.float32)
            pic_t /= 255.
            mask_t = self.mask[i, :, :]
            gt[i, :, :] = pic_t
            meas += np.multiply(mask_t.numpy(), pic_t)
        output_w = self.output_w
        output_h = self.output_h

        frames_hm = np.zeros((self.ratio,self.num_classes, output_h, output_w), dtype=np.float32)
        frames_wh = np.zeros((self.ratio,self.max_objs, 2), dtype=np.float32)
        frames_reg = np.zeros((self.ratio,self.max_objs, 2), dtype=np.float32)
        frames_ind = np.zeros((self.ratio,self.max_objs), dtype=np.int64)
        frames_reg_mask = np.zeros((self.ratio,self.max_objs), dtype=np.uint8)
        
        for frame_id in range(self.ratio):
            num_objs = min(len(frames_bbox_list[frame_id]), self.max_objs)
            hm,wh,reg,reg_mask,ind = self.gen_centernet_label(frames_bbox_list[frame_id],output_h,output_h,num_objs)
            frames_hm[frame_id] = hm
            frames_wh[frame_id] = wh 
            frames_reg[frame_id] = reg 
            frames_reg_mask[frame_id] = reg_mask 
            frames_ind[frame_id] = ind 

            # import matplotlib.pyplot as plt
            # ax1 = plt.subplot(2,1,1)
            # plt.imshow(gt_images_list[frame_id],alpha=1)
            # t = cv2.resize(frames_hm[frame_id][0],(512,512))
            # plt.imshow(t,alpha=0.5)
            # ax2 = plt.subplot(2,1,2)
            # plt.imshow(frames_hm[frame_id][0],alpha=1)
            # plt.show()

        hm = np.zeros((self.num_classes, output_h, output_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)

        meas_label_list = []
        for key in box_dict.keys():
            temp_box = np.array(box_dict[key])
            x1_min = np.min(temp_box[:,0])
            y1_min = np.min(temp_box[:,1])
            x2_max = np.max(temp_box[:,2])
            y2_max = np.max(temp_box[:,3])
            meas_label_list.append([x1_min,y1_min,x2_max,y2_max])
        num_objs = min(len(meas_label_list), self.max_objs)
        hm,wh,reg,reg_mask,ind = self.gen_centernet_label(meas_label_list,output_h,output_h,num_objs)
        
        # import matplotlib.pyplot as plt
        # ax1 = plt.subplot(2,1,1)
        # plt.imshow(meas,alpha=1)
        # ax2 = plt.subplot(2,1,2)
        # plt.imshow(hm[0],alpha=1)
        # plt.show()
        ret = {"input":torch.from_numpy(meas).unsqueeze(0),'hm': hm, 'reg':reg, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh }
        ret.update({"frames_hm":frames_hm,
                    "frames_reg":frames_reg,
                    "frames_reg_mask":frames_reg_mask,
                    "frames_ind":frames_ind,
                    "frames_wh":frames_wh})
        ret.update({"gt_images":np.array(gt_images_list)})
        return ret
    def __len__(self,):
        return len(self.img_files)
    def gen_centernet_label(self,bboxes,output_w,output_h,num_objs):
        hm = np.zeros((self.num_classes, output_h, output_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        for k in range(num_objs):
            bbox = np.array(bboxes[k])

            bbox[[0,2]] = bbox[[0,2]] * output_w
            bbox[[1,3]] = bbox[[1,3]] * output_h
            # bbox = np.clip(bbox, 0, output_res - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if (h > 0 and w > 0):
                
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius)) 
                ct = np.array(
                [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                wh[k] = 1. * w, 1. * h
                
                ind_temp= ct_int[1] * output_w + ct_int[0]
                if ind_temp>=output_h*output_w or ind_temp<0:
                    # print("index:",ind_temp)
                    continue
                ind[k] = ind_temp
                reg[k] = ct - ct_int
                reg_mask[k] = 1

                self.draw_gaussian(hm[0], ct_int, radius)
        return hm,wh,reg,reg_mask,ind


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
            bbox =obj.find("bndbox")
            trackid = int(obj.find("trackid").text)
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
        return len(self.img_files)
            

if __name__=="__main__":
    datasets = VIDData(vid_dir="/media/wangls/new_disk/datasetes/VID2015/imagenet2015/ILSVRC")
    for data in datasets:
        print("hello")
