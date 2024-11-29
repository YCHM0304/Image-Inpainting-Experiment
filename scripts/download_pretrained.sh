#!/bin/bash

# 檢查gdown
if ! command -v gdown &> /dev/null; then
    echo "正在安裝 gdown..."
    pip install --upgrade --no-cache-dir gdown
fi

# 檢查並創建 Pretrained 目錄
if [ ! -d "./Pretrained" ]; then
    mkdir -p "./Pretrained"
fi

# 下載數據
gdown --id 16DJS7TPb5tQ4jiECrCEc84Gj2oOP-MCe --output "./Pretrained/Pretrained.zip"

# 檢查下載是否成功
if [ $? -eq 0 ] && [ -f "./Pretrained/Pretrained.zip" ]; then
    echo "下載成功，開始解壓縮..."
    unzip -q "./Pretrained/Pretrained.zip" -d "./Pretrained"

    if [ $? -eq 0 ]; then
        echo "解壓縮成功"
        mv "./Pretrained/30/generator.pth" "./Pretrained/generator.pth"
        mv "./Pretrained/30/discriminator.pth" "./Pretrained/discriminator.pth"
        rm -r "./Pretrained/30"
        rm "./Pretrained/Pretrained.zip"
        echo "清理完成"
    else
        echo "解壓縮失敗"
        exit 1
    fi
else
    echo "下載失敗"
    exit 1
fi