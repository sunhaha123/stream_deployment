#!/bin/bash

# 切换到 tests 目录
cd tests

# 检查 output.png 是否存在，如果存在则删除
if [ -f "output.png" ]; then
    echo "删除已存在的 output.png 文件"
    rm output.png
fi

# 执行 curl.sh 脚本
sh curl.sh