from Dataset import StainDataset
import json
import os
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torch
from tool import save_image
from tool import rmse
from model import Gen_with_Diffusion, DiffusionModel, BoundaryAwareRefineNet, get_generator

# 定義預處理函數
image_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

mask_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = StainDataset(root_dir='./Data', transform=image_transform, mask_transform=mask_transform)

# read config file
with open('config.json', 'r') as f:
    config = json.load(f)

# load hyperparameters
g_type = config['model']['g_type']
latent_dim = config['model']['latent_dim']
batch_size = config['training']['batch_size']

# 分割train data 和 test data
total_size = len(dataset)
train_size = int(0.8 * total_size)
test_size = total_size - train_size

_, test_dataset = random_split(dataset, [train_size, test_size])

# 初始化Generator、Diffusion Model和RefineNet
base_generator = get_generator(g_type).to(device)
diffusion_model = DiffusionModel(num_steps=5).to(device)
refine_net = BoundaryAwareRefineNet().to(device)
# 整合為新的生成器模型
generator = Gen_with_Diffusion(base_generator, diffusion_model, refine_net).to(device)



# 創建DataLoader
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

if __name__ == '__main__':
    # 檢查是否有pretrained model
    if os.path.exists('./Pretrained/generator.pth'):
        generator.load_state_dict(torch.load('./Pretrained/generator.pth'))
        print("Pretrained model loaded.")
    else:
        print("No pretrained model found.")
        exit()
    with torch.no_grad():
        for i, (real_imgs, stained_imgs, mask) in enumerate(test_loader):  # 包含 mask
            real_imgs = real_imgs.to(device)
            stained_imgs = stained_imgs.to(device)
            mask = mask.to(device)

            # 隨機生成噪聲向量
            z = torch.randn(real_imgs.size(0), latent_dim).to(device)

            # 生成最終修復後的圖像（經過 RefineNet）
            final_imgs = generator(z, stained_imgs, mask)

            RMSE_stained, RMSE_non_stained, RMSE_total = rmse(final_imgs.cpu().numpy(), real_imgs.cpu().numpy(), mask.cpu().numpy())

            # 打印各部分的 RMSE
            print("RMSE in stained region: {:.6f}".format(RMSE_stained))
            print("RMSE in non-stained region: {:.6f}".format(RMSE_non_stained))
            print("Total RMSE: {:.6f}".format(RMSE_total))


            # 拼接 stained_imgs 和 final_imgs 做對比
            comparison = torch.cat((stained_imgs, final_imgs), dim=3)  # 在寬度維度拼接

            # 儲存生成的對比圖像
            save_image(comparison, epoch="Test", iter=i)

            # 僅測試一個批次
            if i == 5:
                break