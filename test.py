import json
import os
from PIL import Image
from tool import rmse, create_comparison
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from model import Discriminator, Gen_with_Diffusion, DiffusionModel, BoundaryAwareRefineNet, get_generator

def tensor_to_image(tensor):
    tensor = tensor.squeeze(0).detach().cpu()
    tensor = (tensor * 0.5 + 0.5).clamp(0, 1)
    return transforms.ToPILImage()(tensor)

# 檢查是否可以使用 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

if __name__ == '__main__':
    # 加載stained mask圖像
    test_data_dir = "./test"
    stained_img_path = os.path.join(test_data_dir, 'stained.jpg')
    mask_img_path = os.path.join(test_data_dir, 'mask.jpg')
    real_img_path = os.path.join(test_data_dir, 'real.jpg')

    # read config file
    with open('config.json', 'r') as f:
        config = json.load(f)

    # load hyperparameters
    g_type = config['model']['g_type']
    latent_dim = config['model']['latent_dim']

    # 加載圖像並預處理
    stained_img = Image.open(stained_img_path).convert('RGB')
    mask_img = Image.open(mask_img_path).convert('L')

    stained_img_tensor = image_transform(stained_img).unsqueeze(0).to(device)
    mask_tensor = mask_transform(mask_img).unsqueeze(0).to(device)

    # 初始化Generator、Diffusion Model和RefineNet
    base_generator = get_generator(g_type).to(device)
    diffusion_model = DiffusionModel(num_steps=5).to(device)
    refine_net = BoundaryAwareRefineNet().to(device)
    discriminator = Discriminator().to(device)

    # 整合為新的生成器模型
    generator = Gen_with_Diffusion(base_generator, diffusion_model, refine_net).to(device)

    # 檢查是否有pretrained model
    if os.path.exists('./Pretrained/generator.pth') and os.path.exists('./Pretrained/discriminator.pth'):
        generator.load_state_dict(torch.load('./Pretrained/generator.pth'))
        discriminator.load_state_dict(torch.load('./Pretrained/discriminator.pth'))
        print("Pretrained model loaded.")
    else:
        print("No pretrained model found.")
        exit()

    # 生成隨機noise向量
    z = torch.randn(1, latent_dim).to(device)

    # 使用生成器生成最终图像
    generator.eval()
    with torch.no_grad():
        generated_img = generator(z, stained_img_tensor, mask_tensor)

    # 加載真實圖像進行對比
    real_img = Image.open(real_img_path).convert('RGB')
    real_img_tensor = image_transform(real_img).unsqueeze(0).to(device)

    # 計算 RMSE
    RMSE_stained, RMSE_non_stained, RMSE_total = rmse(
        generated_img.cpu().detach().numpy(),
        real_img_tensor.cpu().detach().numpy(),
        mask_tensor.cpu().detach().numpy()
    )

    # 打印 RMSE 结果
    print("RMSE in stained region: {:.6f}".format(RMSE_stained))
    print("RMSE in non-stained region: {:.6f}".format(RMSE_non_stained))
    print("Total RMSE: {:.6f}".format(RMSE_total))

    if not os.path.exists('./test_output'):
        os.makedirs('./test_output')

    # 可視化結果
    generated_img_pil = tensor_to_image(generated_img)
    real_img_pil = tensor_to_image(real_img_tensor)
    generated_img_pil.save(os.path.join('./test_output', 'output_generated.png'))
    tensor_to_image(stained_img_tensor).save(os.path.join('./test_output', 'output_stained.png'))

    grid = create_comparison(stained_img_tensor, real_img_tensor, generated_img)

    # Convert to PIL and save
    plt.figure(figsize=(15, 5))
    plt.imshow(grid.permute(1, 2, 0).cpu())
    plt.axis('off')
    plt.savefig(os.path.join('./test_output', 'comparison.png'), bbox_inches='tight', pad_inches=0)
    plt.close()