from torch import nn 
import torch
import torch.nn.functional as F
from models.centernet import get_centernet
from models.admm_net import ADMM_net
def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class AdmmNet(nn.Module):
    def __init__(self,opt):
        super(AdmmNet,self).__init__()
        self.Phi = opt.mask.expand([opt.batch_size, opt.ratio, opt.width, opt.height]).to(opt.device)
        self.Phi_s = opt.mask_s.expand([opt.batch_size, opt.width, opt.height]).to(opt.device)
        self.admm_net = ADMM_net()
        try:
            self.admm_net.load_state_dict(torch.load(opt.admm_net_path))
        except:
            raise("No admm_net pre_train model!")
    def forward(self,meas):
        meas = meas.squeeze()
        return self.admm_net(meas,self.Phi,self.Phi_s)

class ConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,stride=2,padding=1):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,
                                        stride=stride,padding=padding,bias=False)
        self.bn=nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU()

    def forward(self,input):
        x=self.conv1(input)
        return self.relu(self.bn(x))
class SpatialAttention(nn.Module):
    def __init__(self,):
        super(SpatialAttention,self).__init__()
        self.conv = nn.Conv2d(2,1,7,1,padding=3,bias=False)
    def forward(self,x):
        x_mean = torch.mean(x,dim=1,keepdim=True)
        x_max,_ = torch.max(x,dim=1,keepdim=True)
        x_cat = torch.cat([x_mean,x_max],dim=1)
        conv_out = self.conv(x_cat)
        out = F.sigmoid(conv_out)
        return out*x
class ChannelAttention(nn.Module):
    def __init__(self,channels):
        super(ChannelAttention,self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.conv1=nn.Conv2d(channels,channels//2,kernel_size=1)
        self.relu=nn.ReLU()
        self.conv2=nn.Conv2d(channels//2,channels,kernel_size=1)
        self.sigmoid=nn.Sigmoid()
    def forward(self,input):
        avg = self.avgpool(input)
        out = self.relu(self.conv1(avg))
        out = self.sigmoid(self.conv2(out))
        out = input*out
        return out


class CBAM(nn.Module):
    def __init__(self,channels):
        super(CBAM,self).__init__()
        self.channel_attention = ChannelAttention(channels=channels)
        self.spatial_attention = SpatialAttention()
    def forward(self,input):
        out = self.channel_attention(input)
        out = self.spatial_attention(out)
        return out


class FeatureFusionModule(nn.Module):
    def __init__(self,num_classes,in_channels):
        super().__init__()
        self.in_channels=in_channels

        self.convblock=ConvBlock(in_channels=self.in_channels,out_channels
                                                     =num_classes,stride=1)
        self.conv1=nn.Conv2d(num_classes,num_classes,kernel_size=1)
        self.relu=nn.ReLU()
        self.conv2=nn.Conv2d(num_classes,num_classes,kernel_size=1)
        self.sigmoid=nn.Sigmoid()
        self.avgpool=nn.AdaptiveAvgPool2d(output_size=(1,1))

    def forward(self,input_1,input_2):
        x=torch.cat((input_1,input_2),dim=1)
        assert self.in_channels==x.size(1)
        feature=self.convblock(x)

        x=self.avgpool(feature)
        x=self.relu(self.conv1(x))
        x=self.sigmoid(self.conv2(x))
        x=torch.mul(feature,x)

        x=torch.add(x,feature)
        return x
class CombineModel(nn.Module):
    def __init__(self,opt,in_ch,out_chs=[32,64],strides=[2,2],head_conv=256,final_kernel=1):
        super(CombineModel,self).__init__()
        self.centernet = get_centernet(34, opt.heads,head_conv).to(opt.device)
        self.opt = opt
        # if opt.load_model != "":
        #    self.centernet.load_state_dict(torch.load("checkpoints/centernet.pth"))
        self.conv_list1 = nn.ModuleList()
        self.conv_list2 = nn.ModuleList()
        self.conv_last = nn.ModuleList()
        self.cbam_list1 = nn.ModuleList()
        self.num = 8
        self.heads = opt.heads
        last_out = 256
        for i in range(self.num):
            self.conv_list1.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=in_ch,
                        out_channels=out_chs[0],
                        kernel_size=3,
                        stride=strides[0],
                        padding=1),
                    nn.BatchNorm2d(out_chs[0]),
                    nn.ReLU(inplace=True)
                )
            )
            self.conv_list2.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=out_chs[0],
                        out_channels=out_chs[1],
                        kernel_size=3,
                        stride=strides[1],
                        padding=1),
                    nn.BatchNorm2d(out_chs[1]),
                    nn.ReLU(inplace=True)
                )
            )
            # self.conv_last.append(
            #     nn.Sequential(
            #         nn.Conv2d(in_channels=last_out,
            #             out_channels=last_out,
            #             kernel_size=3,
            #             stride=1,
            #             padding=1),
            #     nn.BatchNorm2d(last_out),
            #     nn.ReLU()
            #     )
            # )
            self.conv_last.append(
                FeatureFusionModule(last_out,last_out//2)
            )
            self.cbam_list1.append(
                CBAM(last_out)
            )
            for head in self.heads:
                classes = self.heads[head]
                if head_conv > 0:
                    fc = nn.Sequential(
                        nn.Conv2d(last_out, head_conv,
                            kernel_size=3, padding=1, bias=True),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(head_conv, classes, 
                            kernel_size=final_kernel, stride=1, 
                            padding=final_kernel // 2, bias=True))
                    if 'hm' in head:
                        fc[-1].bias.data.fill_(-2.19)
                    else:
                        fill_fc_weights(fc)
                else:
                    fc = nn.Conv2d(last_out, classes, 
                        kernel_size=3, stride=1, 
                        padding=final_kernel // 2, bias=True)
                    if 'hm' in head:
                        fc.bias.data.fill_(-2.19)
                    else:
                        fill_fc_weights(fc)
                self.__setattr__(str(i)+"_"+head, fc)

    def forward(self,meas,admm_out):
        centernet_out = self.centernet(meas)
        feat_out = centernet_out[0]
        centernet_last_out = centernet_out[1]
        out_list = []
        for i in range(self.num):
            input = admm_out[:,i].unsqueeze(1)
            # temp(input,i-1)
            out = self.conv_list1[i](input)
            # temp(out,i)
            out = self.conv_list2[i](out)
            # temp(out,i+1)
            pred_hm = torch.clamp(F.sigmoid(centernet_last_out["hm"]), min=1e-4, max=1-1e-4)
            # import matplotlib.pyplot as plt
            # plt.imshow(pred_hm.detach().cpu().numpy()[0][0])
            # plt.show()
            out = self.conv_last[i](out,feat_out)*pred_hm
            # temp(out,i+2)
            out = self.cbam_list1[i](out)
            per_frame_out_dict = {}
            for head in self.heads:
                per_frame_out_dict[head] = self.__getattr__(str(i)+"_"+head)(out)
            out_list.append(per_frame_out_dict)
        return centernet_last_out,out_list
count = 0

def temp(out,i):
    import numpy as np 
    from torchvision.utils import make_grid
    feat_vis = out[0].unsqueeze(1)
    size = feat_vis.shape[0]
    feat_vis = make_grid(feat_vis,nrow=int(np.sqrt(size)))
    imshow(feat_vis,i)
def imshow(image,name="temp"):
    global count
    import matplotlib.pyplot as plt
    import cv2
    import os.path as osp
    image = image.detach().cpu().numpy()
    image = image.transpose((1,2,0))[:,:,0]
    plt.imshow(image,cmap="gray")
    # cv2.imshow("image",image)
    # cv2.waitKey(0)
    cv2.imwrite("tt.png",image)
    save_dir = "temp"
    image_name = osp.join(save_dir,str(name))
    print("image_name:",image_name)
    if osp.exists(image_name+".png"):
        image_name = image_name+str(count)
        count+=1
    plt.savefig("{}.png".format(image_name),dpi=300)