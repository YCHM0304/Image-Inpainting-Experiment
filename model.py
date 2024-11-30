import cv2
import functools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_generator(type='concat_conv_deconv'):
  if type == 'concat_conv_deconv':
    return Gen_concat_conv_deconv()
  elif type == 'conv_concat_deconv':
    return Gen_conv_concat_deconv()
  elif type == 'unet-64':
    return UnetGenerator(input_nc=4, output_nc=3, num_downs=6, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False)

class Gen_conv_concat_deconv(nn.Module):
  def __init__(self):
    super().__init__()

    def conv_bn_LRelu(in_dim, out_dim):
      return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_dim),
        nn.LeakyReLU(0.2, inplace=True),
      )

    def deconv_bn_Relu(in_dim, out_dim, output_layer=False):
      if output_layer:
        return nn.Sequential(
          nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
          nn.Tanh()
        )
      return nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(inplace=True),
      )

    self.stained_img_encoder = nn.Sequential(
      conv_bn_LRelu(4, 64),
      conv_bn_LRelu(64, 128),
      conv_bn_LRelu(128, 256),
      nn.Flatten(),
      nn.Linear(256 * 8 * 8, 512)
    )

    self.noise_encoder = nn.Sequential(
      nn.Linear(100, 256),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(256, 512)
    )

    self.combined_model = nn.Sequential(
      nn.Linear(1024, 256 * 8 * 8),  # 新增的線性層
      nn.ReLU(inplace=True),
      nn.Unflatten(1, (256, 8, 8)),  # 重塑為 4D 張量
      deconv_bn_Relu(256, 128),
      deconv_bn_Relu(128, 64),
      deconv_bn_Relu(64, 3, output_layer=True)
    )
  def forward(self, z, stained_imgs, mask):
    combined_input = torch.cat([stained_imgs, mask], dim=1)
    encoded_img = self.stained_img_encoder(combined_input)
    encoded_noise = self.noise_encoder(z)
    concated_noise = torch.cat([encoded_img, encoded_noise], dim=1)
    output = self.combined_model(concated_noise)
    return output  # 重塑輸出成圖像的尺寸

class Gen_concat_conv_deconv(nn.Module):
  def __init__(self):
    super().__init__()

    self.fc = nn.Linear(100, 3 * 64 * 64)
    self.attention = nn.Sequential(
      nn.Conv2d(7, 3, kernel_size=3, padding=1),
      nn.Sigmoid()
    )

    def conv_bn_LRelu(in_dim, out_dim):
      return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_dim),
        nn.LeakyReLU(0.2, inplace=True),
      )

    def deconv_bn_Relu(in_dim, out_dim, output_layer=False):
      if output_layer:
        return nn.Sequential(
          nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
          nn.Tanh()
        )
      return nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(inplace=True),
      )

    self.combined_model = nn.Sequential(
      conv_bn_LRelu(3, 64),
      conv_bn_LRelu(64, 128),
      conv_bn_LRelu(128, 256),
      deconv_bn_Relu(256, 128),
      deconv_bn_Relu(128, 64),
      deconv_bn_Relu(64, 3, output_layer=True)
    )
  def forward(self, z, stained_imgs, mask):
    latent_v = self.fc(z)
    latent_v = latent_v.view(-1, 3, 64, 64)
    combined_input = torch.cat([stained_imgs, mask, latent_v], dim=1)
    attention = self.attention(combined_input)
    combined_input = stained_imgs * (1-attention) + latent_v * attention
    output = self.combined_model(combined_input)
    return output  # 重塑輸出成圖像的尺寸

# 修改生成器以整合擴散模型
class Gen_with_Diffusion(nn.Module):
    def __init__(self, base_generator, diffusion_model, refine_net):
        super(Gen_with_Diffusion, self).__init__()
        self.base_generator = base_generator  # 原本的生成器
        self.diffusion_model = diffusion_model  # 擴散模型
        self.refine_net = refine_net  # Boundary-aware RefineNet

    def forward(self, z, stained_imgs, mask):
        # 使用生成器生成初步影像
        gen_imgs = self.base_generator(z, stained_imgs, mask)

        # 使用擴散模型進一步去噪
        refined_imgs = self.diffusion_model(gen_imgs, mask)

        # 使用 Boundary-aware RefineNet 修復邊界
        final_imgs = self.refine_net(refined_imgs, mask)

        return final_imgs

class ResBlock(nn.Module):
    """Residual Block with two convolutional layers."""
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x):
        return x + self.block(x)

class DiffusionModel(nn.Module):
    def __init__(self, num_steps=5):
        super(DiffusionModel, self).__init__()
        self.num_steps = num_steps

        # Diffusion blocks
        def diffusion_block(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.2, inplace=True)
            )

        # Attention mechanism
        def spatial_attention(dim):
            return nn.Sequential(
                nn.Conv2d(dim, 1, kernel_size=1),
                nn.Sigmoid()
            )

        # Model architecture
        self.initial_block = diffusion_block(4, 64)  # Input: stained_imgs + mask
        self.res_block1 = ResBlock(64)
        self.res_block2 = ResBlock(64)
        self.attention = spatial_attention(64)  # Spatial attention layer
        self.final_block = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, padding=1),  # Output: restored image
        )

    def forward(self, x, mask):
        """
        x: Initial generated image
        mask: Mask indicating regions to restore (single channel)
        """
        # Ensure mask is 4D [batch_size, 1, height, width]
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)

        # Extend mask to match the input shape
        mask = mask.expand(-1, 3, -1, -1)  # Same shape as x

        # Multi-step denoising process
        for step in range(self.num_steps):
            # Dynamically adjust noise level
            noise_level = 0.1 * (1 - (step / self.num_steps))
            noise = torch.randn_like(x) * noise_level

            # Add noise to input
            x_noisy = x + noise

            # Combine input and mask
            combined_input = torch.cat([x_noisy, mask[:, :1, :, :]], dim=1)  # 4-channel input

            # Forward pass through model
            x_restored = self.initial_block(combined_input)
            x_restored = self.res_block1(x_restored)
            x_restored = self.res_block2(x_restored)

            # Apply spatial attention
            attention_weights = self.attention(x_restored)
            x_restored = x_restored * attention_weights

            # Final restoration
            x_restored = self.final_block(x_restored)

            # Update x only in mask regions
            x = x * (1 - mask) + x_restored * mask

        return x

class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, latent_dim=100, num_downs=6, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()

        self.inner_nc = ngf * min(2 ** (num_downs - 1), 8)

        self.latent_processor = nn.Sequential(
            nn.Linear(latent_dim, self.inner_nc),
            nn.ReLU(True)
        )

        # 64x64 -> 32x32
        self.first_conv = nn.Conv2d(input_nc, ngf, kernel_size=4,
                                  stride=2, padding=1, bias=use_bias)

        # 下採樣路徑: 32x32 -> 1x1
        mult = 1
        for i in range(num_downs - 1):
            mult_prev = mult
            mult = min(2 ** (i + 1), 8)

            if i == num_downs - 2:  # 最內層
                down_block = nn.Sequential(
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(ngf * mult_prev, ngf * mult,
                             kernel_size=4, stride=2, padding=1, bias=use_bias)
                )
            else:
                down_block = nn.Sequential(
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(ngf * mult_prev, ngf * mult,
                             kernel_size=4, stride=2, padding=1, bias=use_bias),
                    norm_layer(ngf * mult)
                )
            self.down_layers.append(down_block)

        # 上採樣路徑改為6層: 1x1 -> 64x64
        for i in range(num_downs):  # 注意這裡改為 num_downs
            mult_prev = min(2 ** (num_downs - i - 1), 8)
            mult = min(2 ** (num_downs - i - 2), 8) if i < num_downs - 1 else 1

            if i == 0:  # 最內層: 1x1 -> 2x2
                up_block = nn.Sequential(
                    nn.ReLU(True),
                    nn.ConvTranspose2d(ngf * mult_prev * 2, ngf * mult,
                                     kernel_size=4, stride=2, padding=1, bias=use_bias),
                    norm_layer(ngf * mult)
                )
            elif i == num_downs - 1:  # 最外層: 32x32 -> 64x64
                up_block = nn.Sequential(
                    nn.ReLU(True),
                    nn.ConvTranspose2d(ngf * 2, output_nc,  # 輸入是通道數是ngf*2因為有skip connection
                                     kernel_size=4, stride=2, padding=1)
                )
            else:
                up_block = nn.Sequential(
                    nn.ReLU(True),
                    nn.ConvTranspose2d(ngf * mult_prev * 2, ngf * mult,
                                     kernel_size=4, stride=2, padding=1, bias=use_bias),
                    norm_layer(ngf * mult)
                )
                if use_dropout and i < num_downs - 3:
                    up_block = nn.Sequential(up_block, nn.Dropout(0.5))
            self.up_layers.append(up_block)

    def forward(self, z, stained_imgs, mask):
        features = []

        # 串接輸入
        x = torch.cat([stained_imgs, mask], dim=1)


        # 下採樣路徑
        x = self.first_conv(x)  # 64x64 -> 32x32
        features.append(x)

        for i, down_layer in enumerate(self.down_layers):
            x = down_layer(x)
            features.append(x)

        # 處理latent
        latent_features = self.latent_processor(z)
        latent_features = latent_features.view(-1, self.inner_nc, 1, 1)

        # 串接latent
        x = torch.cat([x, latent_features], 1)

        # 上採樣路徑
        for i, up_layer in enumerate(self.up_layers):
            if i == 0:
                x = up_layer(x)
            else:
                x = up_layer(torch.cat([x, features[-(i+1)]], 1))

        return x

class UnetBD(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super().__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()

        self.first_conv = nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias)

        # 下採樣路徑不變
        mult = 1
        for i in range(num_downs - 1):
            mult_prev = mult
            mult = min(2 ** (i + 1), 8)

            if i == num_downs - 2:
                down_block = nn.Sequential(
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(ngf * mult_prev, ngf * mult, kernel_size=4, stride=2, padding=1, bias=use_bias)
                )
            else:
                down_block = nn.Sequential(
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(ngf * mult_prev, ngf * mult, kernel_size=4, stride=2, padding=1, bias=use_bias),
                    norm_layer(ngf * mult)
                )
            self.down_layers.append(down_block)

        # 修改上採樣路徑，使用 num_downs 而不是 num_downs - 1
        for i in range(num_downs):
            mult_prev = min(2 ** (num_downs - i - 1), 8)
            mult = min(2 ** (num_downs - i - 2), 8) if i < num_downs - 1 else 1

            if i == 0:
                up_block = nn.Sequential(
                    nn.ReLU(True),
                    nn.ConvTranspose2d(ngf * mult_prev, ngf * mult, kernel_size=4, stride=2, padding=1, bias=use_bias),
                    norm_layer(ngf * mult)
                )
            elif i == num_downs - 1:  # 最後一層
                up_block = nn.Sequential(
                    nn.ReLU(True),
                    nn.ConvTranspose2d(ngf * 2, output_nc, kernel_size=4, stride=2, padding=1)
                )
            else:
                up_block = nn.Sequential(
                    nn.ReLU(True),
                    nn.ConvTranspose2d(ngf * mult_prev * 2, ngf * mult, kernel_size=4, stride=2, padding=1, bias=use_bias),
                    norm_layer(ngf * mult)
                )
                if use_dropout and i < num_downs - 3:
                    up_block = nn.Sequential(up_block, nn.Dropout(0.5))
            self.up_layers.append(up_block)

    def forward(self, x):
        features = []

        x = self.first_conv(x)
        features.append(x)

        for down_layer in self.down_layers:
            x = down_layer(x)
            features.append(x)

        for i, up_layer in enumerate(self.up_layers):
            if i == 0:
                x = up_layer(x)
            else:
                x = up_layer(torch.cat([x, features[-(i+1)]], 1))

        return x

class BoundaryAwareRefineNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.refine = UnetBD(5, 3, 6)

    def forward(self, x, mask):
        # 將 mask 轉換為 numpy array 進行處理
        mask_np = mask.cpu().detach().numpy()
        if len(mask_np.shape) == 4:  # 如果是批次數據
            B, C, H, W = mask_np.shape
            penumbra_masks = []

            # 對每個批次的 mask 進行處理
            for b in range(B):
                current_mask = mask_np[b, 0]  # 假設 mask 是單通道的

                # 創建 kernel
                kernel = np.ones((7, 7), np.uint8)

                # 進行膨脹和腐蝕操作
                dilated_mask = cv2.dilate(current_mask, kernel, iterations=1)
                eroded_mask = cv2.erode(current_mask, kernel, iterations=1)

                # 計算半影遮罩
                penumbra_mask = dilated_mask - eroded_mask
                penumbra_masks.append(penumbra_mask)

            # 將處理後的遮罩轉回 tensor
            penumbra_mask = torch.FloatTensor(np.array(penumbra_masks)).unsqueeze(1)

            # 如果使用 GPU，將遮罩移到相應設備
            if x.is_cuda:
                penumbra_mask = penumbra_mask.cuda()

        else:  # 如果不是批次數據
            mask_np = mask_np[0]  # 假設 mask 是單通道的

            # 創建 kernel
            kernel = np.ones((7, 7), np.uint8)

            # 進行膨脹和腐蝕操作
            dilated_mask = cv2.dilate(mask_np, kernel, iterations=1)
            eroded_mask = cv2.erode(mask_np, kernel, iterations=1)

            # 計算半影遮罩
            penumbra_mask = dilated_mask - eroded_mask

            # 將處理後的遮罩轉回 tensor
            penumbra_mask = torch.FloatTensor(penumbra_mask).unsqueeze(0).unsqueeze(0)

            # 如果使用 GPU，將遮罩移到相應設備
            if x.is_cuda:
                penumbra_mask = penumbra_mask.cuda()

        # 將半影遮罩加入到輸入，構成 5 通道
        combined_input = torch.cat([x, penumbra_mask, mask], dim=1)

        # 邊界修復
        refined_output = self.refine(combined_input)

        return refined_output

class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()

    def conv_bn_lrelu(in_dim, out_dim):
      return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, 5, 2, 2),
        nn.BatchNorm2d(out_dim),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.3)
      )

    self.model = nn.Sequential(
        # 輸入圖像尺寸為4 x 64 X 64
      nn.Conv2d(4, 64, 5, 2, 2),
      nn.LeakyReLU(0.2, inplace=True),

      conv_bn_lrelu(64, 128),
      conv_bn_lrelu(128, 256),
      conv_bn_lrelu(256, 512),
      nn.Conv2d(512, 1, 4),
      nn.Sigmoid()
    )

  def forward(self, img, mask):
    # 只在 mask=1 的區域進行判別
    input = torch.cat((img, mask), dim=1)
    y = self.model(input)
    y = y.view(-1, 1)
    return y