#!/bin/bash

# 设置API端点URL
URL="http://localhost:8080/remove-background"

# 设置要上传的图片文件路径
IMAGE_PATH="test_image.png"

# 执行curl命令
curl -X POST \
     -H "Content-Type: multipart/form-data" \
     -F "file=@$IMAGE_PATH" \
     -o "output.png" \
     $URL

echo "背景移除完成，结果保存为 output.png"