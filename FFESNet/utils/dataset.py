import os
import glob
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class PolypDataset(Dataset):
    """
    Dataloader for polyp segmentation tasks
    """
    def __init__(self, image_root, gt_root, trainsize, augmentations):
        # ----------------------------------------------------------------
        # Init train size and augmentations
        # ----------------------------------------------------------------
        self.trainsize = trainsize
        self.augmentations = augmentations
        
        # ----------------------------------------------------------------
        # Load images and masks
        # ----------------------------------------------------------------
        self.images = glob.glob(os.path.join(image_root, '*.jpg')) +\
                      glob.glob(os.path.join(image_root, '*.png'))
        self.images = sorted(self.images)
        
        self.gts = glob.glob(os.path.join(gt_root, '*.jpg')) +\
                   glob.glob(os.path.join(gt_root, '*.png'))
        self.gts = sorted(self.gts)

        if len(self.images) != len(self.gts):
            raise "Mismatch number of images and gts"
        
        self.filter_files()
        
        self.size = len(self.images)

        # ----------------------------------------------------------------
        # Transform images and masks
        # ----------------------------------------------------------------        
        if self.augmentations == True:
            print('Using RandomRotation, RandomFlip')
            self.img_transform = transforms.Compose([
                transforms.RandomRotation(90, expand=False, center=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0, hue=0),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
            self.gt_transform = transforms.Compose([
                transforms.RandomRotation(90, expand=False, center=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])
            
        else:
            print('No augmentation')
            self.img_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
            
            self.gt_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])
            

    def __getitem__(self, index):
        """Get item method"""
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        
        seed = np.random.randint(2147483647)
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.img_transform is not None:
            image = self.img_transform(image)
            
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.gt_transform is not None:
            gt = self.gt_transform(gt)

        if os.path.basename(self.images[index]) != os.path.basename(self.gts[index]):
            print(self.images[index], self.gts[index])

        return image, gt

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts


    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')


    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


class TestDataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = glob.glob(os.path.join(image_root, '*.jpg')) +\
                      glob.glob(os.path.join(image_root, '*.png'))
        self.images = sorted(self.images)
        
        self.gts = glob.glob(os.path.join(gt_root, '*.jpg')) +\
                   glob.glob(os.path.join(gt_root, '*.png'))
        self.gts = sorted(self.gts)

        if len(self.images) != len(self.gts):
            raise "Mismatch number of images and gts"
        
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
        

# ========================================================================
# Test dataloader
# ========================================================================
def main():
    train_images_path = '/home/ptn/Storage/FFESNet/data/Kvasir_Splited/trainSplited/images'
    train_masks_path = '/home/ptn/Storage/FFESNet/data/Kvasir_Splited/trainSplited/masks'
    trainDataset = PolypDataset(train_images_path, train_masks_path, trainsize=384, augmentations = True)
    train_loader = DataLoader(dataset=trainDataset,batch_size=32,shuffle=True)

    print(trainDataset.size)

    iter_X = iter(train_loader)
    steps_per_epoch = len(iter_X)
    num_epoch = 0
    n_epochs = 10
    total_steps = (n_epochs+1)*steps_per_epoch

    from tqdm import tqdm
    for step in tqdm(range(1, total_steps)):
        # Reset iterators for each epoch
        if step % steps_per_epoch == 0:
            iter_X = iter(train_loader)
            num_epoch = num_epoch + 1

        images, masks = next(iter_X)

if __name__ == "__main__":
    main()