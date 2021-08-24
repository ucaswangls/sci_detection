from torch.utils.data import Dataset 
import numpy as np
import os
import os.path as osp
import torch
import cv2
# from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian 
from utils.auguments import transforms,test_transforms
import math
from utils.ttf_functions import bbox_areas, calc_region

class TrainData(Dataset):
    def __init__(self,opt):
        self.data_dir= opt.train_data_dir
        self.data_list = os.listdir(opt.train_data_dir)
        self.img_files = []
        for image_dir in os.listdir(opt.train_data_dir):
            temp_list = os.listdir(osp.join(opt.train_data_dir,image_dir,"img1"))
            temp_list.sort(key=lambda x:int(x[:-4]))
            temp_list = temp_list[:len(temp_list)//2]
            for tt in range(5):
                meas_list = []
                frame_count = 0
                for i,image_name in enumerate(temp_list):
                    if i % (tt+1) != 0:
                        continue
                    meas_list.append(osp.join(opt.train_data_dir,image_dir,"img1",image_name))
                    frame_count +=1
                    if frame_count%opt.ratio==0:
                        self.img_files.append(meas_list)
                        meas_list = []
                        frame_count=0
        self.mask = opt.mask
        self.ratio,self.resize_h,self.resize_w = self.mask.shape
        self.output_h,self.output_w = self.resize_h//opt.down_ratio,self.resize_w//opt.down_ratio
        self.opt = opt
        self.num_classes = opt.num_classes
        self.max_objs = opt.max_objs
        self.draw_gaussian = draw_umich_gaussian

        self.wh_area_process = "log"
        self.wh_agnostic = True 
        self.wh_gaussian=True
        self.alpha = 0.54 
        self.beta = 0.54 
        self.hm_weight = 1
        self.wh_weight = 5
        self.base_loc = None
        self.wh_planes = 4 if self.wh_agnostic else 4 * self.num_fg
        self.down_ratio = opt.down_ratio

    def __getitem__(self,index):
        gt = np.zeros([self.ratio, self.resize_h, self.resize_w],dtype=np.float32)
        meas = np.zeros([self.resize_h, self.resize_w],dtype=np.float32)
        box_dict = {}
        frames_bbox_list = []
        gt_images_list = []

        image = cv2.imread(self.img_files[index][0])
        im_h,im_w,_ = image.shape
        transform = transforms(im_h=im_h,im_w=im_w)
        meas_bboxes = []
        for i,image_path in enumerate(self.img_files[index]):
            label_path = image_path.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt')
            with open(label_path,"r") as fp:
                lines = fp.readlines()
            image = cv2.imread(image_path)
            per_frame_bbox_list = []
            category_ids = []
            for line in lines:
                bbox = line.strip("\n").split(" ")[1:] 
                id,x,y,w,h = [float(i) for i in bbox]
                x -= w/2
                y -= h/2
                id = int(id)
                # cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255))
                bbox = np.array([x,y,x+w,y+h])
                bbox = np.clip(bbox,a_min=0,a_max=1)
                x1,y1,x2,y2 = bbox
                if x1>=x2 or y1>=y2:
                    continue
                per_frame_bbox_list.append(bbox.tolist())
                category_ids.append(id)
            #transform
            transformed = transform(image=image, bboxes=per_frame_bbox_list,category_ids=category_ids)
            image = transformed["image"]
            per_frame_bbox_list = transformed["bboxes"]

            per_frame_id_list = transformed["category_ids"]

            image = cv2.resize(image,(self.resize_w,self.resize_h))
            for id_index,id in enumerate(per_frame_id_list):
                im_w,im_h,_ = image.shape
                temp_bbox = per_frame_bbox_list[id_index]
                x1 = temp_bbox[0]*im_w
                y1 = temp_bbox[1]*im_h
                x2 = temp_bbox[2]*im_w
                y2 = temp_bbox[3]*im_h
                per_frame_bbox_list[id_index] = [x1,y1,x2,y2]
                meas_bboxes.append([x1,y1,x2,y2])
            #     x1,y1,x2,y2 = [int(i) for i in per_frame_bbox_list[id_index]]
            #     cv2.rectangle(image,(x1,y1),(x2,y2),(255,0,255))
            # import matplotlib.pyplot as plt
            # plt.imshow(image)
            # plt.show()

            frames_bbox_list.append(per_frame_bbox_list)
            pic_t = cv2.cvtColor(image,cv2.COLOR_BGR2YCrCb)[:,:,0]
            gt_images_list.append(pic_t)
            pic_t = pic_t.astype(np.float32)
            pic_t /= 255.
            mask_t = self.mask[i, :, :]
            gt[i, :, :] = pic_t
            meas += np.multiply(mask_t.numpy(), pic_t)
        output_w = self.output_w
        output_h = self.output_h

        frames_hm = torch.zeros(self.ratio,self.num_classes, output_h, output_w)
        frames_bboxes= torch.zeros(self.ratio,4,output_h,output_w)
        frames_weight= torch.zeros(self.ratio,1,output_h,output_w)
        
        for frame_id in range(self.ratio):
            bboxes = np.array(frames_bbox_list[frame_id])
            num_bbox = bboxes.shape[0]
            if  num_bbox != 0:
                labels = np.ones(num_bbox,dtype=np.int32)
                heatmap, box_target, reg_weight = self.target_generator(
                        torch.from_numpy(bboxes),torch.from_numpy(labels))
                frames_hm[frame_id] = heatmap
                frames_bboxes[frame_id] = box_target
                frames_weight[frame_id] = reg_weight 


            # import matplotlib.pyplot as plt
            # ax1 = plt.subplot(2,1,1)
            # plt.imshow(gt_images_list[frame_id],alpha=1)
            # t = cv2.resize(frames_hm[frame_id].numpy()[0],(512,512))
            # print(torch.max(frames_hm[frame_id]))
            # plt.imshow(t,alpha=0.5)
            # ax2 = plt.subplot(2,1,2)
            # plt.imshow(frames_bboxes[frame_id].numpy()[2],alpha=1)
            # plt.show()
        meas_bboxes = np.array(meas_bboxes)
        num_bbox = meas_bboxes.shape[0]
        if num_bbox != 0:
            meas_labels = np.ones(num_bbox,dtype=np.int32)
            heatmap, box_target, reg_weight = self.target_generator(
                    torch.from_numpy(meas_bboxes),torch.from_numpy(meas_labels))
        else:
            heatmap = torch.zeros(self.num_classes, output_h, output_w)
            box_target = torch.zeros(4,output_h,output_w)
            reg_weight = torch.zeros(1,output_h,output_w)
        # import matplotlib.pyplot as plt
        # ax1 = plt.subplot(2,1,1)
        # plt.imshow(meas,alpha=1)
        # ax2 = plt.subplot(2,1,2)
        # plt.imshow(heatmap[0],alpha=1)
        # plt.show()
        ret = {"input":torch.from_numpy(meas).unsqueeze(0)}
        # ret.update({'hm': hm, 'reg':reg, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh })
        ret.update({"frames_hm":frames_hm,
                    "frames_bboxes":frames_bboxes,
                    "frames_reg_weight":frames_weight})
        ret.update({"gt_images":np.array(gt_images_list)})
        ret.update({"heatmap":heatmap})
        ret.update({"box_target":box_target})
        ret.update({"reg_weight":reg_weight})
        return ret
    def target_generator(self, gt_boxes, gt_labels):
        """

        Args:
            gt_boxes: list(tensor). tensor <=> image, (gt_num, 4).
            gt_labels: list(tensor). tensor <=> image, (gt_num,).
            img_metas: list(dict).

        Returns:
            heatmap: tensor, (batch, 80, h, w).
            box_target: tensor, (batch, 4, h, w) or (batch, 80 * 4, h, w).
            reg_weight: tensor, same as box_target.
        """
        feat_shape=(self.output_h,self.output_w)
        heatmap, box_target, reg_weight =self.target_single_image(
            gt_boxes,
            gt_labels,
            feat_shape=feat_shape
        )

        # heatmap, box_target = [torch.stack(t, dim=0).detach() for t in [heatmap, box_target]]
        # reg_weight = torch.stack(reg_weight, dim=0).detach()

        return heatmap, box_target, reg_weight
    def target_single_image(self, gt_boxes, gt_labels, feat_shape):
        """

        Args:
            gt_boxes: tensor, tensor <=> img, (num_gt, 4).
            gt_labels: tensor, tensor <=> img, (num_gt,).
            feat_shape: tuple.

        Returns:
            heatmap: tensor, tensor <=> img, (80, h, w).
            box_target: tensor, tensor <=> img, (4, h, w) or (80 * 4, h, w).
            reg_weight: tensor, same as box_target
        """
        output_h, output_w = feat_shape
        heatmap_channel = self.num_classes

        heatmap = gt_boxes.new_zeros((heatmap_channel, output_h, output_w))
        fake_heatmap = gt_boxes.new_zeros((output_h, output_w))
        box_target = gt_boxes.new_ones((self.wh_planes, output_h, output_w)) * -1
        reg_weight = gt_boxes.new_zeros((self.wh_planes // 4, output_h, output_w))

        if self.wh_area_process == 'log':
            boxes_areas_log = bbox_areas(gt_boxes).log()
        elif self.wh_area_process == 'sqrt':
            boxes_areas_log = bbox_areas(gt_boxes).sqrt()
        else:
            boxes_areas_log = bbox_areas(gt_boxes)
        boxes_area_topk_log, boxes_ind = torch.topk(boxes_areas_log, boxes_areas_log.size(0))

        if self.wh_area_process == 'norm':
            boxes_area_topk_log[:] = 1.

        gt_boxes = gt_boxes[boxes_ind]
        gt_labels = gt_labels[boxes_ind]

        feat_gt_boxes = gt_boxes / self.down_ratio
        feat_gt_boxes[:, [0, 2]] = torch.clamp(feat_gt_boxes[:, [0, 2]], min=0,
                                                max=output_w - 1)
        feat_gt_boxes[:, [1, 3]] = torch.clamp(feat_gt_boxes[:, [1, 3]], min=0,
                                                max=output_h - 1)
        feat_hs, feat_ws = (feat_gt_boxes[:, 3] - feat_gt_boxes[:, 1],
                            feat_gt_boxes[:, 2] - feat_gt_boxes[:, 0])

        # we calc the center and ignore area based on the gt-boxes of the origin scale
        # no peak will fall between pixels
        ct_ints = (torch.stack([(gt_boxes[:, 0] + gt_boxes[:, 2]) / 2,
                                (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2],
                                dim=1) / self.down_ratio).to(torch.int)

        h_radiuses_alpha = (feat_hs / 2. * self.alpha).int()
        w_radiuses_alpha = (feat_ws / 2. * self.alpha).int()
        if self.wh_gaussian and self.alpha != self.beta:
            h_radiuses_beta = (feat_hs / 2. * self.beta).int()
            w_radiuses_beta = (feat_ws / 2. * self.beta).int()

        if not self.wh_gaussian:
            # calculate positive (center) regions
            r1 = (1 - self.beta) / 2
            ctr_x1s, ctr_y1s, ctr_x2s, ctr_y2s = calc_region(gt_boxes.transpose(0, 1), r1)
            ctr_x1s, ctr_y1s, ctr_x2s, ctr_y2s = [torch.round(x.float() / self.down_ratio).int()
                                                    for x in [ctr_x1s, ctr_y1s, ctr_x2s, ctr_y2s]]
            ctr_x1s, ctr_x2s = [torch.clamp(x, max=output_w - 1) for x in [ctr_x1s, ctr_x2s]]
            ctr_y1s, ctr_y2s = [torch.clamp(y, max=output_h - 1) for y in [ctr_y1s, ctr_y2s]]

        # larger boxes have lower priority than small boxes.
        for k in range(boxes_ind.shape[0]):
            cls_id = gt_labels[k] - 1

            fake_heatmap = fake_heatmap.zero_()
            self.draw_truncate_gaussian(fake_heatmap, ct_ints[k],
                                        h_radiuses_alpha[k].item(), w_radiuses_alpha[k].item())
            heatmap[cls_id] = torch.max(heatmap[cls_id], fake_heatmap)

            if self.wh_gaussian:
                if self.alpha != self.beta:
                    fake_heatmap = fake_heatmap.zero_()
                    self.draw_truncate_gaussian(fake_heatmap, ct_ints[k],
                                                h_radiuses_beta[k].item(),
                                                w_radiuses_beta[k].item())
                box_target_inds = fake_heatmap > 0
            else:
                ctr_x1, ctr_y1, ctr_x2, ctr_y2 = ctr_x1s[k], ctr_y1s[k], ctr_x2s[k], ctr_y2s[k]
                box_target_inds = torch.zeros_like(fake_heatmap, dtype=torch.uint8)
                box_target_inds[ctr_y1:ctr_y2 + 1, ctr_x1:ctr_x2 + 1] = 1
            if self.wh_agnostic:
                box_target[:, box_target_inds] = gt_boxes[k][:, None]
                cls_id = 0
            else:
                box_target[(cls_id * 4):((cls_id + 1) * 4), box_target_inds] = gt_boxes[k][:, None]

            if self.wh_gaussian:
                local_heatmap = fake_heatmap[box_target_inds]
                ct_div = local_heatmap.sum()
                local_heatmap *= boxes_area_topk_log[k]
                reg_weight[cls_id, box_target_inds] = local_heatmap / ct_div
            else:
                reg_weight[cls_id, box_target_inds] = \
                    boxes_area_topk_log[k] / box_target_inds.sum().float()
        return heatmap, box_target, reg_weight
    def gaussian_2d(self, shape, sigma_x=1, sigma_y=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]

        h = np.exp(-(x * x / (2 * sigma_x * sigma_x) + y * y / (2 * sigma_y * sigma_y)))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    def draw_truncate_gaussian(self, heatmap, center, h_radius, w_radius, k=1):
        h, w = 2 * h_radius + 1, 2 * w_radius + 1
        sigma_x = w / 6
        sigma_y = h / 6
        gaussian = self.gaussian_2d((h, w), sigma_x=sigma_x, sigma_y=sigma_y)
        gaussian = heatmap.new_tensor(gaussian)

        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape[0:2]

        left, right = min(x, w_radius), min(width - x, w_radius + 1)
        top, bottom = min(y, h_radius), min(height - y, h_radius + 1)

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[h_radius - top:h_radius + bottom,
                          w_radius - left:w_radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        return heatmap
    def __len__(self,):
        return len(self.img_files)
    

class TestData(Dataset):
    def __init__(self,opt):
        self.data_dir= opt.test_data_dir
        self.data_list = os.listdir(opt.test_data_dir)
        self.img_files = []
        for image_dir in os.listdir(opt.test_data_dir):
            temp_list = os.listdir(osp.join(opt.test_data_dir,image_dir,"img1"))
            temp_list.sort(key=lambda x:int(x[:-4]))
            meas_list = []
            for i,image_name in enumerate(temp_list):
                meas_list.append(osp.join(opt.test_data_dir,image_dir,"img1",image_name))
                if (i+1)%8==0:
                    self.img_files.append(meas_list)
                    meas_list = []
        # self.label_files = [x.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt')
        #                     for x in self.img_files]
        self.mask = opt.mask
        self.ratio,self.resize_h,self.resize_w = self.mask.shape
        self.opt = opt
        self.num_classes = opt.num_classes
        self.max_objs = 32
        self.transforms = test_transforms(512,512)
    def __getitem__(self,index):
        gt = np.zeros([self.ratio, self.resize_h, self.resize_w],dtype=np.float32)
        meas = np.zeros([self.resize_h, self.resize_w],dtype=np.float32)
        gt_images_list = []
        for i,image_path in enumerate(self.img_files[index]):
            image = cv2.imread(image_path)
            im_h,im_w = image.shape[:2]
            if im_h <512 or im_w <512:
                image = cv2.resize(image,(512,512))
            transformed = self.transforms(image=image)
            image = transformed["image"]
            image = cv2.resize(image,(self.resize_w,self.resize_h))
            pic_t = cv2.cvtColor(image,cv2.COLOR_BGR2YCrCb)[:,:,0]
            gt_images_list.append(pic_t)
            pic_t = pic_t.astype(np.float32)
            pic_t /= 255.
            mask_t = self.mask[i, :, :]
            gt[i, :, :] = pic_t
            meas += np.multiply(mask_t.numpy(), pic_t)
       
        ret = {"input":torch.from_numpy(meas).unsqueeze(0)}
        ret.update({"gt_images":np.array(gt_images_list)})
        return ret
    def __len__(self,):
        return len(self.img_files)
    
   


