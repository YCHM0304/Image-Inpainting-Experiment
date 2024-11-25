import numpy as np
import matplotlib.pyplot as plt
import math
import os

def save_image(gen_imgs, epoch, iter, show=False):
    """將生成的圖像保存成一張圖"""
    gen_imgs = gen_imgs.cpu().detach().numpy()
    gen_imgs = (gen_imgs + 1) / 2  # 假設生成圖像範圍在 [-1, 1]，先歸一化到 [0, 1]

    # 將圖像裁剪到 [0, 1] 範圍，防止 imshow 超出範圍
    gen_imgs = np.clip(gen_imgs, 0, 1)

    # 計算行數和列數
    n = gen_imgs.shape[0]
    cols = int(math.sqrt(n))
    rows = math.ceil(n / cols)

    fig, axs = plt.subplots(rows, cols, figsize=(10, 10))
    cnt = 0
    for i in range(rows):
        for j in range(cols):
            if cnt < n:
                ax = axs[i, j] if rows > 1 else axs[j]
                ax.imshow(np.transpose(gen_imgs[cnt], (1, 2, 0)))
                ax.axis('off')
                cnt += 1
            else:
                ax = axs[i, j] if rows > 1 else axs[j]
                ax.axis('off')

    if not os.path.exists('./val_imgs'):
        os.makedirs('./val_imgs')
    plt.savefig(f'./val_imgs/Epoch_{epoch}_iter_{iter}.png')
    if show:
        plt.show()
    else:
        plt.close()

def rmse(predictions, targets, mask):
    return np.sqrt((((predictions - targets) * mask) ** 2).mean()), \
    np.sqrt((((predictions - targets) * (1 - mask)) ** 2).mean()), \
    np.sqrt(((predictions - targets) ** 2).mean())