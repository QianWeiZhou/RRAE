from PIL import Image
from io import BytesIO
import torch


class Blade_dataset(torch.utils.data.Dataset):
    def __init__(self, file_path='/home/work/tp/NearLoss/data/Blade/processed/train.pt', transform=None, target_transform=None, img_size=512): 
        super(Blade_dataset,self).__init__()
        self.imgs, self.labels = torch.load(file_path)
        self.transform = transform
        self.target_transform = target_transform
        
        
 
    def __getitem__(self, index):
        blade = Image.open(BytesIO(self.imgs[index])).convert('L')        
        blade_name = self.labels[index]

        if self.transform is not None:
            blade = self.transform(blade)


        return blade, blade_name
 
    def __len__(self): 
        num = len(self.imgs)
        return num
