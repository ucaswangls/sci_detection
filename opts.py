import argparse 
from utils.utils import get_masks

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",type=int,default=200)
    parser.add_argument("--batch_size",type=int,default=8)
    parser.add_argument("--height",type=int,default=512)
    parser.add_argument("--width",type=int,default=512)
    parser.add_argument("--num_workers",type=int,default=4)
    parser.add_argument("--down_ratio",type=int,default=4)
    parser.add_argument("--ratio",type=int,default=8)
    parser.add_argument("--num_classes",type=int,default=1)
    parser.add_argument("--max_objs",type=int,default=400)
    parser.add_argument("--lr",type=float,default=0.0001)
    parser.add_argument("--log_dir",type=str,default="log")
    parser.add_argument("--save_model_step",type=int,default=2)
    parser.add_argument("--show_step",type=int,default=20)
    parser.add_argument("--device",type=str,default="cuda")
    parser.add_argument("--amp",type=bool,default=False)
    parser.add_argument("--test_flag",type=bool,default=None)
    parser.add_argument("--checkpoint",type=str,default=None)
    parser.add_argument("--checkpoints_dir",type=str,default="checkpoints")
    parser.add_argument("--train_data_dir",type=str,default="/home/wangls/datasets/MOT17/images/train")
    parser.add_argument("--test_data_dir",type=str,default="/home/wangls/datasets/MOT17/images/test")
    # parser.add_argument("--train_data_dir",type=str,default="/home/wangls/datasets/MOT_HEAD/HT21/images/train")
    # parser.add_argument("--test_data_dir",type=str,default="/home/wangls/datasets/MOT_HEAD/HT21/images/test")
    parser.add_argument("--mask_path",type=str,default="/home/wangls/datasets/SCI/mask/mask_512.npy")
    parser.add_argument("--admm_net_path",type=str,default="weights/admmnet_53.pth")
    parser.add_argument('--hm_weight', type=float, default=1,
                                help='loss weight for keypoint heatmaps.')
    parser.add_argument('--off_weight', type=float, default=1,
                                help='loss weight for keypoint local offsets.')
    parser.add_argument('--wh_weight', type=float, default=0.1,
                                help='loss weight for bounding box size.')
    args = parser.parse_args()

    args.heads = {'hm': args.num_classes, 'wh': 4}
    args.heads.update({'reg': 2})
    args.mask,args.mask_s = get_masks(args.mask_path)
    args.colors = [[0, 0, 0], [128, 64, 128], [244, 35, 232], [170, 70, 70], [102, 102, 156], [10, 153, 153],
                [153, 20, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35]]

    return args 
