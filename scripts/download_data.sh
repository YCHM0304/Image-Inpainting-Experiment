#!/bin/bash

# 檢查unzip
if ! command -v unzip &> /dev/null; then
   echo "正在安裝 unzip..."
   if command -v apt-get &> /dev/null; then
       sudo apt-get update
       sudo apt-get install -y unzip
   elif command -v yum &> /dev/null; then
       sudo yum install -y unzip
   elif command -v brew &> /dev/null; then
       brew install unzip
   else
       echo "無法自動安裝 unzip，請手動安裝"
       exit 1
   fi
fi

# 檢查gdown
if ! command -v gdown &> /dev/null; then
   echo "正在安裝 gdown..."
   pip install --upgrade --no-cache-dir gdown
fi

# 下載數據
gdown --id 1aGNqHgyDV50KFwdlfBj4tws8HAdABw83 --output "./Data.zip"

# 檢查下載是否成功
if [ $? -eq 0 ] && [ -f "./Data.zip" ]; then
   echo "下載成功，開始解壓縮..."
   unzip -q "./Data.zip" -d "./"

   if [ $? -eq 0 ]; then
       echo "解壓縮成功"
       rm "./Data.zip"
       echo "清理完成"
   else
       echo "解壓縮失敗"
       exit 1
   fi
else
   echo "下載失敗"
   exit 1
fi