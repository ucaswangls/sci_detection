from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from opts import get_args

import numpy as np 
import cv2
import os.path as osp
import torch
# from datasets.vid_dataset import TrainData
from datasets.mot_dataset import TrainData
from torch.utils.data import DataLoader
# from network import get_pose_net
from models.combine_model import CombineModel, AdmmNet
import matplotlib.pyplot as plt
from utils.ttf_functions import ttf_decode
def eval(opt):
    train_data = TrainData(opt)
    # print(opt)
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Creating model...')
    model = CombineModel(opt,1).to(opt.device)
    if opt.checkpoint is not None:
        model.load_state_dict(torch.load(osp.join(opt.checkpoints_dir,opt.checkpoint)))
    model.eval()
    admm_net = AdmmNet(opt).to(opt.device)
    admm_net = admm_net.eval()
    train_loader = DataLoader(
            train_data, 
            batch_size=opt.batch_size, 
            shuffle=True,
            num_workers=opt.num_workers,
            # pin_memory=True,
            drop_last=True
    )
    print('\nStarting training...')
    for epoch in range(1, opt.epochs + 1):
        for iter, batch_data in enumerate(train_loader):
            for k in batch_data:
                batch_data[k] = batch_data[k].to(device=opt.device, non_blocking=True)   
            image = batch_data["input"]
            gt_images = batch_data["gt_images"]
            results_list = []
            with torch.no_grad():
                admm_out = admm_net(image)[-1]
                out_all = model(image,admm_out)[1]
                for out in out_all:
                    gt_image = gt_images[0]
                    # out['hm'] = torch.sigmoid(out['hm'])
                    # scores,bboxes = multi_pose_decode(out["hm"],out["wh"],out["reg"])
                    results = ttf_decode(out["hm"],out["wh"])
                    results_list.append(results)
            image = gt_image
            image = image.squeeze(1).cpu().numpy()
            hm = out['hm']
            hm = hm.squeeze(1).cpu().numpy()

            image = image.transpose(1,0,2).reshape(image.shape[1],-1)
            hm = hm.transpose(1,0,2).reshape(hm.shape[1],-1)

            src_hm = batch_data["heatmap"]
            src_hm = src_hm.squeeze(1).cpu().numpy()
            src_hm = src_hm.transpose(1,0,2).reshape(src_hm.shape[1],-1)

            hm1 = batch_data["frames_hm"][:,0]
            hm1 = hm1.squeeze(1).cpu().numpy()
            hm1 = hm1.transpose(1,0,2).reshape(hm1.shape[1],-1)

            # ax1 = plt.subplot(1,1,1)
            # fig = plt.figure(0)
            # ax = plt.gca()
            count = 0
           
            image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
            for i in range(len(results_list)):
                for j,bboxes in enumerate(results_list[i][0][0]):
                    score = bboxes[4]
                    if score.item() < 0.2:
                        continue
                    count += 1
                    box = bboxes[:4]
                    x1,y1,x2,y2 = [int(x) for x in box]
                    box[[0,2]] = box[[0,2]] + i*512
                    x1,y1,x2,y2 = [int(ii) for ii in box]
                    # 默认框的颜色是黑色，第一个参数是左上角的点坐标
                    # 第二个参数是宽，第三个参数是长
                    # ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, color="blue", fill=False, linewidth=1))
                    rand_index = np.random.randint(0,len(opt.colors))
                    cv2.rectangle(image,(x1,y1),(x2,y2),opt.colors[rand_index],thickness=2)
            cv2.imwrite("results/{}_{}.png".format(epoch,iter),image)
            # plt.imshow(image,cmap="gray")
            # plt.show()
            # plt.savefig("results/{}_{}.png".format(epoch,iter),dpi=300)
            # plt.close(0)

if __name__ == '__main__':
    opt =  get_args()
    eval(opt)
