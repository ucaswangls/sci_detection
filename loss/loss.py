from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn 
from loss.loss_function import FocalLoss, RegL1Loss
from utils.utils import _sigmoid 

class CenterNetLoss(nn.Module):
    def __init__(self, opt):
        super(CenterNetLoss, self).__init__()
        self.opt  = opt
        self.crit = FocalLoss()
        self.crit_reg = RegL1Loss()

    def forward(self,output, gt_label,hm_flag=False):
        opt = self.opt
        hm_loss, wh_loss, off_loss , = 0, 0, 0, 
        # output = outputs[0]
        output['hm'] = _sigmoid(output['hm'])
        hm_loss += self.crit(output['hm'], gt_label['hm']) 
        if hm_flag:
            return hm_loss
        if opt.wh_weight > 0:
            #宽高
            wh_loss += self.crit_reg(output['wh'], gt_label['reg_mask'],
                                    gt_label['ind'], gt_label['wh']) 
            
        if opt.off_weight > 0:
            #中心点回归
            off_loss += self.crit_reg(output['reg'], gt_label['reg_mask'],
                                    gt_label['ind'], gt_label['reg']) 
        loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + \
            opt.off_weight * off_loss 
        return loss
