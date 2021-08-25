from datasets.mot_dataset import TrainData
# from datasets.vid_dataset import TrainData
from torch.utils.data import DataLoader
from models.combine_model import CombineModel,AdmmNet
from utils.utils import Logger,save_image
import torch.optim as optim
import torch
from torch.utils.tensorboard import SummaryWriter
from opts import get_args
import os
import os.path as osp
from loss.ttf_loss import TTFLoss 
from torch.cuda.amp import autocast,GradScaler
import time

def train(args,network,logger,writer=None):
    admm_net = AdmmNet(args).to(args.device)
    admm_net = admm_net.eval()
    #创建数据集
    dataset = TrainData(args)
    train_data_loader = DataLoader(dataset=dataset, 
            batch_size=args.batch_size, 
            shuffle=True,
            drop_last=True,
            num_workers=args.num_workers)
    #创建优化器
    optimizer = optim.Adam(network.parameters(), lr=args.lr)
    #学习率衰减
    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.9)
    #损失判别器
    # criterion = CenterNetLoss(args)
    criterion = TTFLoss().to(args.device)
    #amp
    scaler = GradScaler()
    for epoch in range(args.epochs):
        #保存模型
        if epoch % args.save_model_step == 0:
            if not osp.exists(args.checkpoints_dir):
                os.makedirs(args.checkpoints_dir)
            torch.save(network.state_dict(),osp.join(args.checkpoints_dir,"combine_model_"+str(epoch)+".pth"))
        #训练
        epoch_loss = 0
        network = network.train()
        for iteration, batch_data in enumerate(train_data_loader):
            for k in batch_data:
                batch_data[k] = batch_data[k].to(device=args.device)   
            meas = batch_data["input"]
            optimizer.zero_grad()
            #forward
            if args.amp:
                with autocast():
                    with torch.no_grad():
                        admm_out = admm_net(meas)[-1]
                    out = network(meas,admm_out)
            else: 
                with torch.no_grad():
                    admm_out = admm_net(meas)[-1]
                    # if True:
                    #     if iteration % 100 == 0:
                    #         sing_out = admm_out[0].cpu().numpy()
                    #         gt = batch_data["gt_images"]
                    #         sing_gt = gt[0].cpu().numpy()/255.
                    #         image_name = "temp/"+str(iteration)+".png"
                    #         save_image(sing_out,sing_gt,image_name)
                out = network(meas,admm_out) 
            #loss
            # loss1 = criterion(out[0],batch_data)
            aux_out = out[0]
            pred_heatmap = aux_out["hm"]
            pred_wh= aux_out["wh"]
            heatmap = batch_data["heatmap"]
            box_target = batch_data["box_target"]
            reg_weight = batch_data["reg_weight"]
            loss1 = criterion(pred_heatmap,pred_wh,heatmap,box_target,reg_weight)
            
            frames_loss = 0
            for i,frame_out in enumerate(out[1]):
                # gt_data = {}
                # gt_data["hm"] = batch_data["frames_hm"][:,i]
                # gt_data["reg_mask"] = batch_data["frames_reg_mask"][:,i]
                # gt_data["reg"] = batch_data["frames_reg"][:,i]
                # gt_data["wh"] = batch_data["frames_wh"][:,i]
                # gt_data["ind"] = batch_data["frames_ind"][:,i]
                # gt_bboxes = batch_data["frames_bboxes"][:,i]
                # gt_labels = batch_data["frames_labels"][:,i]
                pred_heatmap = frame_out["hm"]
                pred_wh= frame_out["wh"]
                heatmap = batch_data["frames_hm"][:,i]
                box_target= batch_data["frames_bboxes"][:,i]
                reg_weight= batch_data["frames_reg_weight"][:,i]
                frames_loss += criterion(pred_heatmap,pred_wh,heatmap,box_target,reg_weight)
                # frames_loss += criterion(frame_out,gt_data)
            
            sum_loss = 0.1*loss1+frames_loss
            epoch_loss += sum_loss
            #backward
            if args.amp:
                scaler.scale(sum_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                sum_loss.backward()
                optimizer.step()

            #损失可视化
            if iteration % args.show_step == 0:
                lr = optimizer.state_dict()["param_groups"][0]["lr"]
                logger.info("epoch: {}, iter: {}, lr: {:.6f}, loss: {:.3f}.".format(epoch,iteration,lr,sum_loss.item()))
                writer.add_scalar("loss",sum_loss.item(),epoch*len(train_data_loader) + iteration)
        #学习率衰减
        scheduler.step()
        #mean loss
        logger.info("epoch: {}, mean_loss: {:.3f}.".format(epoch,epoch_loss.item()/(iteration+1)))

if __name__ == '__main__':
    args = get_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    #创建日志目录
    log_dir = osp.join(args.log_dir,"log")
    show_dir = osp.join(args.log_dir,"show")
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    if not osp.exists(show_dir):
        os.makedirs(show_dir)
    logger = Logger(log_dir)
    writer = SummaryWriter(log_dir = show_dir)

    #创建网络 
    network = CombineModel(args,in_ch=1)
    network = network.to(args.device)
    #加载模型
    if args.checkpoint is not None:
        logger.info("Load pre_train model...")
        network.load_state_dict(torch.load(osp.join(args.checkpoints_dir,args.checkpoint)))
    else:
        logger.info("No pre_train model")
    #开始训练
    train(args,network,logger,writer)
    writer.close()
