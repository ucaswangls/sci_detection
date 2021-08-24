from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from datasets.mot_dataset import TestData
from opts import get_args 
from torch.utils.data import DataLoader
from models.combine_model import CombineModel,AdmmNet
import matplotlib.pyplot as plt
import os.path as osp
from utils.ttf_functions import ttf_decode
def test(opt):
    admm_net = AdmmNet(opt).to(opt.device)
    admm_net = admm_net.eval()
    test_data = TestData(opt)
    # print(opt)
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Creating model...')
    # model = get_pose_net(34, opt.heads, opt.head_conv).to(opt.device)
    model = CombineModel(opt,1).to(opt.device)
    #创建优化器
    start_epoch = 0
    if opt.checkpoint is not None:
        model.load_state_dict(torch.load(osp.join(opt.checkpoints_dir,opt.checkpoint)))
    model.eval()
    test_loader = DataLoader(
            test_data, 
            batch_size=opt.batch_size, 
            shuffle=True,
            num_workers=opt.num_workers,
            # pin_memory=True,
            drop_last=True
    )
    for epoch in range(start_epoch + 1, opt.epochs + 1):
        for iter, batch_data in enumerate(test_loader):
            for k in batch_data:
                batch_data[k] = batch_data[k].to(device=opt.device, non_blocking=True)   
            meas = batch_data["input"]
            gt_images = batch_data["gt_images"]
            with torch.no_grad():
                admm_out = admm_net(meas)[-1]
                out = model(meas,admm_out)[1][0]
                gt_image = gt_images[:,0]
                # out['hm'] = torch.sigmoid(out['hm'])
                # scores,bboxes = multi_pose_decode(out["hm"],out["wh"],out["reg"])
                results = ttf_decode(out["hm"],out["wh"])
            image = gt_image
            image = image.squeeze(1).cpu().numpy()
            hm = out['hm']
            hm = hm.squeeze(1).cpu().numpy()

            image = image.transpose(1,0,2).reshape(image.shape[1],-1)
            hm = hm.transpose(1,0,2).reshape(hm.shape[1],-1)

            ax1 = plt.subplot(2,1,1)
            ax = plt.gca()
            count = 0
            for i in range(len(results)):
                for j,bboxes in enumerate(results[i][0]):
                    score = bboxes[4]
                    if score.item() < 0.1:
                        continue
                    count += 1
                    box = bboxes[:4]
                    x1,y1,x2,y2 = [int(x) for x in box]
                    box[[0,2]] = box[[0,2]] + i*512
                    x1,y1,x2,y2 = [int(ii) for ii in box]
                    # 默认框的颜色是黑色，第一个参数是左上角的点坐标
                    # 第二个参数是宽，第三个参数是长
                    ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, color="blue", fill=False, linewidth=1))
            print("count1:",count)
            plt.imshow(image,cmap="gray")
            # plt.show()
            ax3 = plt.subplot(2,1,2)
            plt.imshow(hm)
            plt.show()
            print("epoch:",epoch)

if __name__ == '__main__':
    opt = get_args()
    test(opt)