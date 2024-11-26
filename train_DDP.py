import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import json
from tool import save_image
from model import Discriminator, Gen_with_Diffusion, DiffusionModel, BoundaryAwareRefineNet, get_generator
from loss import AdversarialLoss, BoundaryLoss
from Dataset import StainDataset, image_transform, mask_transform
from torch.utils.data import DataLoader, random_split

def setup(rank, world_size):
    """
    初始化進程組
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """
    清理進程組
    """
    dist.destroy_process_group()

def main(rank, world_size, config):
    """
    主訓練函數
    """
    # 設置分散式環境
    setup(rank, world_size)

    # 從配置文件讀取參數
    batch_size = config['training']['batch_size']
    epochs = config['training']['epochs']
    sample_interval = config['training']['sample_interval']
    g_type = config['model']['g_type']
    latent_dim = config['model']['latent_dim']
    lambda_l1 = config['loss_weights']['lambda_l1']
    lambda_bd = config['loss_weights']['lambda_bd']
    lr_g = config['optimizer']['lr_g']
    lr_d = config['optimizer']['lr_d']

    # 數據集設置
    dataset = StainDataset(root_dir='./Data',
                          transform=image_transform,
                          mask_transform=mask_transform)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # 創建分散式採樣器和數據加載器
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size // world_size,  # 將batch size分配到每個GPU
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )

    # 初始化模型
    base_generator = get_generator(g_type).to(rank)
    diffusion_model = DiffusionModel(num_steps=5).to(rank)
    refine_net = BoundaryAwareRefineNet().to(rank)
    generator = Gen_with_Diffusion(base_generator, diffusion_model, refine_net).to(rank)
    discriminator = Discriminator().to(rank)

    # 包裝成DDP模型
    generator = DDP(generator, device_ids=[rank], find_unused_parameters=True)
    discriminator = DDP(discriminator, device_ids=[rank], find_unused_parameters=True)

    # 初始化損失函數和優化器
    adversarial_loss = AdversarialLoss()
    l1_loss = torch.nn.L1Loss()
    bd_loss = BoundaryLoss()

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))

    # 訓練循環
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)

        if rank == 0:
            progress_bar = tqdm(train_loader,
                              desc=f'Epoch [{epoch+1}/{epochs}]')
        else:
            progress_bar = train_loader

        for i, (real_imgs, stained_imgs, mask) in enumerate(progress_bar):
            # 將數據移到對應的GPU
            real_imgs = real_imgs.to(rank)
            stained_imgs = stained_imgs.to(rank)
            mask = mask.to(rank)
            batch_size = real_imgs.size(0)

            # 生成標籤
            valid = torch.ones(batch_size, 1, requires_grad=False).to(rank)
            fake = torch.zeros(batch_size, 1, requires_grad=False).to(rank)

            # 訓練生成器
            optimizer_G.zero_grad()
            z = torch.randn(batch_size, latent_dim).to(rank)
            gen_imgs = generator(z, stained_imgs, mask)

            g_adv_loss = adversarial_loss(discriminator(gen_imgs, mask), valid)
            g_l1_loss = l1_loss(gen_imgs, real_imgs)
            g_bd_loss = bd_loss(gen_imgs, stained_imgs, real_imgs, mask)
            g_loss = g_adv_loss + lambda_l1 * g_l1_loss + lambda_bd * g_bd_loss

            g_loss.backward()
            optimizer_G.step()

            # 訓練判別器
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(real_imgs, mask), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), mask), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            # 只在主進程(rank 0)更新進度條和保存結果
            if rank == 0:
                progress_bar.set_postfix({
                    'D_loss': f'{d_loss.item():.4f}',
                    'G_loss': f'{g_loss.item():.4f}'
                })

                if i % sample_interval == 0:
                    save_image(gen_imgs, epoch+1, i)

        # 在每個epoch結束時保存模型（只在主進程）
        if rank == 0:
            checkpoint = {
                'epoch': epoch,
                'generator_state_dict': generator.module.state_dict(),
                'discriminator_state_dict': discriminator.module.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
            }
            torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pt')

    cleanup()

if __name__ == "__main__":
    # 讀取配置文件
    with open('config.json', 'r') as f:
        config = json.load(f)

    # 獲取可用的GPU數量
    world_size = torch.cuda.device_count()
    print(f'Using {world_size} GPUs!')

    # 使用 spawn 啟動多進程
    mp.spawn(
        main,
        args=(world_size, config),
        nprocs=world_size,
        join=True
    )