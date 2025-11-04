import torch
import torch.utils.data as data
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import torchvision
import os,sys
sys.path.append('.')
sys.path.append('..')
import random
from PIL import Image
random.seed(1143)


class Train_dataset(data.Dataset):
    def __init__(self,path,train,size=240,format='.png'):
        super(Train_dataset,self).__init__()
        self.size=size
        print('crop size',size)
        self.train=train
        self.format=format
        haze_imgs_dir=os.listdir(os.path.join(path,'raw_256_256'))
        self.haze_imgs_dir = [x for x in haze_imgs_dir if ('.png' in x)]
        self.haze_imgs = [os.path.join(path,'raw_256_256',img) for img in self.haze_imgs_dir]
        self.reference_dir = os.path.join(path, 'depth_train')
        self.clear_dir=os.path.join(path,'gt_256_256')


    def __getitem__(self, index):

        haze=Image.open(self.haze_imgs[index])
        if isinstance(self.size,int):
            while haze.size[0]<self.size or haze.size[1]<self.size or ('.png' not in self.haze_imgs[index]):
                index=random.randint(0,699)
                haze=Image.open(self.haze_imgs[index])

        img=self.haze_imgs[index]
        index1 = random.randint(0, 699)
        haze2 = Image.open((self.haze_imgs[index1]))
        name_syn=img.split('/')[-1]
        id = name_syn
        clear_name=id
        clear=Image.open(os.path.join(self.clear_dir, clear_name))
        clear=tfs.CenterCrop(haze.size[::-1])(clear)
        reference = Image.open(os.path.join(self.reference_dir, clear_name))
        reference = tfs.ToTensor()(reference)

        if not isinstance(self.size,str) and self.train:
            i,j,h,w=tfs.RandomCrop.get_params(haze,output_size=(self.size,self.size))
            haze=FF.crop(haze,i,j,h,w)
            clear=FF.crop(clear,i,j,h,w)
            haze2 = FF.crop(haze2, i, j, h, w)
        haze123,clear=self.augData([haze, haze2],clear.convert("RGB") )
        return haze123, clear, reference

    def augData(self,data,target):

        rand_hor=random.randint(0,1)
        rand_rot=random.randint(0,3)
        img1 = data[0]
        img2 = data[1]
        img1 = tfs.RandomHorizontalFlip(rand_hor)(img1)
        img2 = tfs.RandomHorizontalFlip(rand_hor)(img2)
        target=tfs.RandomHorizontalFlip(rand_hor)(target)
        if rand_rot:
            img1 = FF.rotate(img1,90*rand_rot)
            img2 = FF.rotate(img2, 90 * rand_rot)
            target=FF.rotate(target,90*rand_rot)

        img1 = tfs.ToTensor()(img1)
        img2 = tfs.ToTensor()(img2)
        data = torch.cat([img1, img2], dim=0)
        target = tfs.ToTensor()(target)
        return data ,target
    def __len__(self):
        return len(self.haze_imgs)

class Val_kdataset(data.Dataset):
    def __init__(self,path,train,size=240,format='.png'):
        super(Val_kdataset,self).__init__()
        self.size=size
        print('crop size',size)
        self.train=train
        self.format=format
        haze_imgs_dir=os.listdir(os.path.join(path,'raw_256_256'))
        self.haze_imgs_dir = [x for x in haze_imgs_dir if ('.png' in x)]
        self.haze_imgs = [os.path.join(path,'raw_256_256',img) for img in self.haze_imgs_dir]
        self.clear_dir=os.path.join(path,'gt_256_256')
        self.reference_dir = os.path.join(path, 'depth_val')


    def __getitem__(self, index):
        haze=Image.open(self.haze_imgs[index])
        if isinstance(self.size,int):
            while haze.size[0]<self.size or haze.size[1]<self.size or ('.png' not in self.haze_imgs[index]):
                index=random.randint(0,89)
                haze=Image.open(self.haze_imgs[index])

        img=self.haze_imgs[index]
        name_syn=img.split('/')[-1]
        id = name_syn
        clear_name=id
        clear=Image.open(os.path.join(self.clear_dir,clear_name))
        clear=tfs.CenterCrop(haze.size[::-1])(clear)
        reference = Image.open(os.path.join(self.reference_dir, clear_name))
        reference = tfs.ToTensor()(reference)
        haze,clear=self.augData(haze.convert("RGB") ,clear.convert("RGB") )
        return haze, clear, reference
    def augData(self,data,target):
        data = data.resize((256, 256))
        target = target.resize((256, 256))
        data=tfs.ToTensor()(data)
        target=tfs.ToTensor()(target)
        return  data ,target
    def __len__(self):
        return len(self.haze_imgs)
