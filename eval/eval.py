from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import torch
from data import TrainData
from opts import opts
from torch.utils.data import DataLoader
# from network import get_pose_net
from combine_model import CombineModel
import matplotlib.pyplot as plt
from utils.decode import multi_pose_decode
from models import ADMM_net
def train(opt):
    train_data = TrainData(opt)
    # print(opt)
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Creating model...')
    # model = get_pose_net(34, opt.heads, opt.head_conv).to(opt.device)
    model = CombineModel(opt,1,heads=opt.heads).to(opt.device)
    #创建优化器
    start_epoch = 0
    if opt.load_model is not None:
        model.load_state_dict(torch.load(opt.load_model))
    model = torch.nn.DataParallel(model)
    model.eval()
    admm_net = ADMM_net().to(opt.device)
    admm_net = admm_net.eval()
    admm_net.load_state_dict(torch.load("weights/admmnet_76.pth"))
    train_loader = DataLoader(
            train_data, 
            batch_size=4, 
            shuffle=True,
            num_workers=opt.num_workers,
            # pin_memory=True,
            drop_last=True
    )
    print('\nStarting training...')
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        for iter, batch_data in enumerate(train_loader):
            for k in batch_data:
                batch_data[k] = batch_data[k].to(device=opt.device, non_blocking=True)   
            image = batch_data["input"]
            mask = opt.mask.to(opt.device)
            mask_s = opt.mask_s.to(opt.device)
            frames_hm = batch_data["frames_hm"]
            batch_size,frames,classes,height,width = frames_hm.shape
            height *= opt.down_ratio
            width *= opt.down_ratio
            Phi = mask.expand([batch_size, frames, width, height])
            Phi_s = mask_s.expand([batch_size, width, height])
            meas_t = image.squeeze(1)
            with torch.no_grad():
                admm_out = admm_net(meas_t,Phi,Phi_s)

                out = model(image,admm_out[-1])[0]
                
                out['hm'] = torch.sigmoid(out['hm'])
                scores,bboxes = multi_pose_decode(out["hm"],out["wh"],out["reg"])
            image = image.squeeze(1).cpu().numpy()
            hm = out['hm']
            hm = hm.squeeze(1).cpu().numpy()

            image = image.transpose(1,0,2).reshape(image.shape[1],-1)
            hm = hm.transpose(1,0,2).reshape(hm.shape[1],-1)

            src_hm = batch_data["hm"]
            src_hm = src_hm.squeeze(1).cpu().numpy()
            src_hm = src_hm.transpose(1,0,2).reshape(src_hm.shape[1],-1)

            ax1 = plt.subplot(3,1,1)
            
            ax = plt.gca()
            count = 0
            for i in range(scores.shape[0]):
                for j,score in enumerate(scores[i]):
                    if score.item() < 0.1:
                        continue
                    count += 1
                    box = bboxes[i][j]
                    box[[0,2]] = box[[0,2]] + i*128
                    x1,y1,x2,y2 = [int(ii/128.*512) for ii in box]

                    # 默认框的颜色是黑色，第一个参数是左上角的点坐标
                    # 第二个参数是宽，第三个参数是长
                    ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, color="blue", fill=False, linewidth=1))
            print("count1:",count)
            plt.imshow(image)
            ax2 = plt.subplot(3,1,2)
            plt.imshow(src_hm)
            # plt.show()
            ax3 = plt.subplot(3,1,3)
            # ax2 = plt.gca()

            # count = 0
            # for i in range(scores.shape[0]):
            #     for j,score in enumerate(scores[i]):
            #         if score.item() < 0.1:
            #             continue
            #         count += 1
            #         box = bboxes[i][j]
            #         box[[0,2]] = box[[0,2]] + i*128
            #         x1,y1,x2,y2 = [int(ii) for ii in box]

            #         # 默认框的颜色是黑色，第一个参数是左上角的点坐标
            #         # 第二个参数是宽，第三个参数是长
            #         ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, color="blue", fill=False, linewidth=1))
            # print("count:",count)
            plt.imshow(hm)
            plt.show()
            print("epoch:",epoch)

if __name__ == '__main__':
    opt = opts().parse()
    train(opt)