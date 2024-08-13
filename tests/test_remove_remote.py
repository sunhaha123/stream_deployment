import requests
import time
import os
from pathlib import Path

# 设置API端点URL
url = "https://internal-ems.echo.tech/removebg/remove-background"

# 设置要上传的图片文件路径
image_path = "test_image.png"

# 设置输出文件路径
output_path = Path("output.png")

# 如果output.png存在，则删除它
if output_path.exists():
    os.remove(output_path)
    print(f"已删除已存在的 {output_path}")

# 开始计时
start_time = time.time()

# 准备文件
files = {'file': open(image_path, 'rb')}

# 发送POST请求
response = requests.post(url, files=files)

# 检查请求是否成功
if response.status_code == 200:
    # 保存结果
    output_path.write_bytes(response.content)
    print(f"背景移除完成，结果保存为 {output_path}")
else:
    print(f"请求失败，状态码：{response.status_code}")

# 结束计时并计算耗时
end_time = time.time()
total_time = end_time - start_time

print(f"整体耗时：{total_time:.2f} 秒")