from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from tool import save_image
from model import Discriminator, Gen_with_Diffusion, DiffusionModel, BoundaryAwareRefineNet, get_generator
from loss import AdversarialLoss, BoundaryLoss
from Dataset import StainDataset, image_transform, mask_transform
from torch.utils.data import DataLoader, random_split
import json

# -----------------------------------
#           Hyperparameters
# -----------------------------------

# read config file
with open('config.json', 'r') as f:
    config = json.load(f)

# load hyperparameters
epochs = config['training']['epochs']
batch_size = config['training']['batch_size']
sample_interval = config['training']['sample_interval']
g_type = config['model']['g_type']
latent_dim = config['model']['latent_dim']
lambda_l1 = config['loss_weights']['lambda_l1']
lambda_bd = config['loss_weights']['lambda_bd']
lr_g = config['optimizer']['lr_g']
lr_d = config['optimizer']['lr_d']

# ----------------------------------
#           Dataset setup
# ----------------------------------

# 創建數據集
dataset = StainDataset(root_dir='./Data', transform=image_transform, mask_transform=mask_transform)


# 分割train data 和 test data
total_size = len(dataset)
train_size = int(0.8 * total_size)
test_size = total_size - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 創建DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ----------------------------------
#           Training setup
# ----------------------------------

# 檢查是否可以使用 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化Generator、Diffusion Model和RefineNet
base_generator = get_generator(g_type).to(device)
diffusion_model = DiffusionModel(num_steps=5).to(device)
refine_net = BoundaryAwareRefineNet().to(device)
discriminator = Discriminator().to(device)

# 整合為新的生成器模型
generator = Gen_with_Diffusion(base_generator, diffusion_model, refine_net).to(device)

# 將模型調成訓練模式
generator.train()
discriminator.train()

# 初始化Loss function
adversarial_loss = AdversarialLoss()  # 對抗損失(只計算mask範圍)
l1_loss = torch.nn.L1Loss()  # L1損失(Global)
bd_loss = BoundaryLoss()  # Boundary-aware損失

# 設定優化器
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))


if __name__ == "__main__":
    # 開始訓練
    for epoch in range(epochs):
        progress_bar = tqdm(train_loader)
        last_gen_imgs = None  # 儲存最後一個生成的批次

        for i, (real_imgs, stained_imgs, mask) in enumerate(progress_bar):

            # 將資料移動到 GPU 上
            real_imgs, stained_imgs, mask = real_imgs.to(device), stained_imgs.to(device), mask.to(device)
            batch_size = real_imgs.size(0)

            # 創建標籤
            valid = torch.ones(batch_size, 1, requires_grad=False).to(device)  # 真實標籤
            fake = torch.zeros(batch_size, 1, requires_grad=False).to(device)  # 假標籤

            # ---- 訓練生成器 ----
            optimizer_G.zero_grad()

            # 隨機生成噪聲並生成圖像
            z = torch.randn(batch_size, latent_dim).to(device)
            gen_imgs = generator(z, stained_imgs, mask)
            last_gen_imgs = gen_imgs  # 儲存最後一個批次生成的圖像

            # 計算生成器損失（對抗損失 + 加權的 L1 損失）
            g_adv_loss = adversarial_loss(discriminator(gen_imgs, mask), valid)
            g_l1_loss = l1_loss(gen_imgs, real_imgs)
            g_bd_loss = bd_loss(gen_imgs, stained_imgs, real_imgs, mask)

            g_loss = g_adv_loss + lambda_l1 * g_l1_loss + lambda_bd * g_bd_loss
            # g_loss = g_l1_loss + lambda_bd * g_bd_loss


            # 反向傳播並更新生成器的權重
            g_loss.backward()
            optimizer_G.step()

            # ---- 訓練鑑別器 ----
            optimizer_D.zero_grad()

            # 計算鑑別器損失（對真實圖像和假圖像）
            real_loss = adversarial_loss(discriminator(real_imgs, mask), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), mask), fake)
            d_loss = (real_loss + fake_loss) / 2

            # 反向傳播並更新鑑別器的權重
            d_loss.backward()
            optimizer_D.step()

            # 更新進度條資訊
            progress_bar.set_postfix({
                'Loss_D': round(d_loss.item(), 4),
                'Loss_G': round(g_loss.item(), 4),
                'Epoch': epoch + 1,
            })

            # 每隔一定批次儲存生成的圖像
            if i % sample_interval == 0:
                print(f"Epoch [{epoch + 1}/{epochs}] Batch {i}/{len(train_loader)} \ Loss D: {d_loss.item():.4f}, Loss G: {g_loss.item():.4f}")
                save_image(gen_imgs, epoch+1, i)

        # 每個 epoch 儲存最後一批次的生成圖像
        if last_gen_imgs is not None:
            save_image(last_gen_imgs, epoch+1, "latest")

    torch.save(generator.state_dict(), 'generator.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')
