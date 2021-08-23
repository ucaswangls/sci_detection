import torch
import scipy.io as scio
import numpy as np
from torch import nn 
import logging 
import time 
import os 
import os.path as osp
import cv2

def get_masks(mask_path):
    mask = np.load(mask_path)
    mask_s = np.sum(mask,axis=0)
    mask_s[mask_s==0] = 1
    return torch.from_numpy(mask,),torch.from_numpy(mask_s)

def save_image(out,gt,image_name,show_flag=False):
    sing_out = out.transpose(1,0,2).reshape(out.shape[1],-1)
    sing_gt = gt.transpose(1,0,2).reshape(gt.shape[1],-1)
    result_img = np.concatenate([sing_out,sing_gt],axis=0)*255
    cv2.imwrite(image_name,result_img)
    if show_flag:
        cv2.namedWindow("image",0)
        cv2.imshow("image",result_img.astype(np.uint8))
        cv2.waitKey(0)
    
class double_conv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(double_conv, self).__init__()
        self.d_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.d_conv(x)
        return x

def generate_masks(mask_path):
    mask = scio.loadmat(osp.join(mask_path,'mask.mat'))
    mask = mask['mask']
    mask = np.transpose(mask, [2, 0, 1])
    mask_s = np.sum(mask, axis=0,dtype=np.float32)
    mask_s[mask_s==0] = 1
    mask = torch.from_numpy(mask)
    mask = mask.float()
    mask_s = torch.from_numpy(mask_s)
    mask_s = mask_s.float()
    return mask, mask_s

def A(x,Phi):
    temp = x*Phi
    y = torch.sum(temp,1)
    return y

def At(y,Phi):
    temp = torch.unsqueeze(y, 1).repeat(1,Phi.shape[1],1,1)
    x = temp*Phi
    return x

def Logger(log_dir):
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(filename)s [line: %(lineno)s] - %(message)s")

    localtime = time.strftime("%Y_%m_%d_%H_%M_%S")
    logfile = osp.join(log_dir,localtime+".log")
    fh = logging.FileHandler(logfile,mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger 

import numba
@numba.jit(nopython=True, nogil=True)
def gen_oracle_map(feat, ind, w, h):
  # feat: B x maxN x featDim
  # ind: B x maxN
  batch_size = feat.shape[0]
  max_objs = feat.shape[1]
  feat_dim = feat.shape[2]
  out = np.zeros((batch_size, feat_dim, h, w), dtype=np.float32)
  vis = np.zeros((batch_size, h, w), dtype=np.uint8)
  ds = [(0, 1), (0, -1), (1, 0), (-1, 0)]
  for i in range(batch_size):
    queue_ind = np.zeros((h*w*2, 2), dtype=np.int32)
    queue_feat = np.zeros((h*w*2, feat_dim), dtype=np.float32)
    head, tail = 0, 0
    for j in range(max_objs):
      if ind[i][j] > 0:
        x, y = ind[i][j] % w, ind[i][j] // w
        out[i, :, y, x] = feat[i][j]
        vis[i, y, x] = 1
        queue_ind[tail] = x, y
        queue_feat[tail] = feat[i][j]
        tail += 1
    while tail - head > 0:
      x, y = queue_ind[head]
      f = queue_feat[head]
      head += 1
      for (dx, dy) in ds:
        xx, yy = x + dx, y + dy
        if xx >= 0 and yy >= 0 and xx < w and yy < h and vis[i, yy, xx] < 1:
          out[i, :, yy, xx] = f
          vis[i, yy, xx] = 1
          queue_ind[tail] = xx, yy
          queue_feat[tail] = f
          tail += 1
  return out
def get_masks(mask_path):
    mask = np.load(mask_path)
    mask_s = np.sum(mask,axis=0)
    mask_s[mask_s==0] = 1
    return torch.from_numpy(mask,),torch.from_numpy(mask_s)

def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y

def pre_process(opt, image, scale, meta=None):
    height, width = image.shape[0:2]
    new_height = int(height * scale)
    new_width  = int(width * scale)

    inp_height, inp_width = opt.input_h, opt.input_w
    c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
    s = max(height, width) * 1.0
    # trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    resized_image = cv2.resize(image, (inp_width, inp_height))
    # inp_image = cv2.warpAffine(
    #   resized_image, trans_input, (inp_width, inp_height),
    #   flags=cv2.INTER_LINEAR)
    inp_image = ((resized_image / 255. - opt.mean) / opt.std).astype(np.float32)
    images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
    images = torch.from_numpy(images)
    meta = {'c': c, 's': s, 
            'out_height': inp_height // opt.down_ratio, 
            'out_width': inp_width // opt.down_ratio}
    return images, meta

def transfrom_out_wls(dets,w,h,s_w,s_h):
  dets[:,0] = dets[:,0]/w*s_w
  dets[:,1] = dets[:,1]/h*s_h
  print(s_w)
  print(s_h)
  return dets
def multi_pose_post_process(dets, c, s, h, w):
  # dets: batch x max_dets x 40
  # return list of 39 in image coord
  ret = []
  for i in range(dets.shape[0]):
    # bbox = transform_preds(dets[i, :, :4].reshape(-1, 2), c[i], s[i], (w, h))
    # pts = transform_preds(dets[i, :, 5:9].reshape(-1, 2), c[i], s[i], (w, h))
    # bbox = transfrom_out_wls(dets[i, :, :4].reshape(-1, 2),w=256,h=128,s_w=1920,s_h=1080)
    # pts = transfrom_out_wls(dets[i, :, 5:9].reshape(-1, 2),w=256,h=128,s_w=1920,s_h=1080)
    bbox = transfrom_out_wls(dets[i, :, :4].reshape(-1, 2),w=w,h=h,s_w=int(c[0][0]*2),s_h=int(c[0][1]*2))
    pts = transfrom_out_wls(dets[i, :, 5:9].reshape(-1, 2),w=w,h=h,s_w=int(c[0][0]*2),s_h=int(c[0][1]*2))
    top_preds = np.concatenate(
      [bbox.reshape(-1, 4), dets[i, :, 4:5], 
       pts.reshape(-1, 4),dets[i,:,9:]], axis=1).astype(np.float32).tolist()
    ret.append({np.ones(1, dtype=np.int32)[0]: top_preds})
  return ret
def post_process(num_classes, dets, meta, scale=1):
    temp_long = dets.shape[2]
    dets = dets.detach().cpu().numpy().reshape(1, -1, temp_long)
    dets = multi_pose_post_process(
      dets.copy(), [meta['c']], [meta['s']],
      meta['out_height'], meta['out_width'])
    for j in range(1, num_classes + 1):
      dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, temp_long)
      # import pdb; pdb.set_trace()
      dets[0][j][:, :4] /= scale
      dets[0][j][:, 5:] /= scale
    return dets[0]
def merge_outputs(detections):
    results = {}
    results[1] = np.concatenate(
        [detection[1] for detection in detections], axis=0).astype(np.float32)
    results[1] = results[1].tolist()
    return results
