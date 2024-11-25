from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os

class StainDataset(Dataset):
    def __init__(self, root_dir, transform=None, mask_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.original_path = os.path.join(root_dir, 'Original')
        self.stained_path = os.path.join(root_dir, 'Stained')
        self.mask_path = os.path.join(root_dir, 'Mask')
        self.images = os.listdir(self.original_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        original_img_path = os.path.join(self.original_path, img_name)
        stained_img_path = os.path.join(self.stained_path, img_name)
        mask_img_path = os.path.join(self.mask_path, img_name)

        original_image = Image.open(original_img_path)
        stained_image = Image.open(stained_img_path)
        mask_image = Image.open(mask_img_path).convert('L')  # 讀取為灰階圖像

        if self.transform:
            original_image = self.transform(original_image)
            stained_image = self.transform(stained_image)

        if self.mask_transform:
            mask_image = self.mask_transform(mask_image)

        return original_image, stained_image, mask_image

# 定義轉換
image_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

mask_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),  # 將單通道 mask 轉為 tensor，但不進行 Normalize
])