import random 
import albumentations as A
from albumentations.augmentations.crops.transforms import CenterCrop 

def transforms(im_h,im_w,scales=[0.35,0.6],):
    p = random.random()
    if p<0.4:
        crop_p=0
    else:
        crop_p = 1
    p = random.random()
    if p<0.5:
        flip_p=0
    else:
        flip_p = 1
    a = int(im_h*scales[0])
    b = int(im_h*scales[1])
    random_h = random.randint(a,b)
    a = int(im_w*scales[0])
    b = int(im_w*scales[1])
    random_w = random.randint(a,b)
    x_min = random.randint(0,im_w-random_w)
    y_min = random.randint(0,im_h-random_h)
    # print("rand_w:",random_w)
    # print("rand_h:",random_h)
    x_max = x_min+random_w
    y_max = y_min+random_h
    transform = A.Compose([ 
        # A.RandomResizedCrop(height=128,width=128,scale=(0.25,0.8),p=0.7),
        A.Crop(x_min=x_min,y_min=y_min,x_max=x_max,y_max=y_max,p=crop_p),
        A.HorizontalFlip(p=flip_p),
        # A.ShiftScaleRotate(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.3),
        ],
        A.BboxParams(format="albumentations",label_fields=['category_ids'],min_visibility=0.1))
    return transform

def test_transforms(im_h,im_w):
    transform = A.Compose([ 
        # A.RandomResizedCrop(height=128,width=128,scale=(0.25,0.8),p=0.7),
        A.CenterCrop(height=im_h,width=im_w),
        ])
    return transform