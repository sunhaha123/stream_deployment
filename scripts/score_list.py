import os
import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# 定义目标文件夹路径
folder_path = "/mnt/new/pivix/pivix_sum_0918"

# 定义接口URL
url = "http://127.0.0.1:8080/compute-score"

# 最大线程数
max_workers = 5  # 可以根据你的需求调整

# 处理单张图片的函数
def process_image(filename):
    img_path = os.path.join(folder_path, filename)

    # 对应的 txt 文件路径
    txt_filename = os.path.splitext(filename)[0] + ".txt"
    txt_path = os.path.join(folder_path, txt_filename)

    # 构造请求数据
    data = {
        "img_path": img_path
    }

    try:
        # 发送POST请求到接口，计算reward
        response = requests.post(url, json=data)

        # 解析响应
        result = response.json()

        if "score" in result:
            reward = float(result["score"])
            print(f"图片: {filename}, 得分: {reward}")

            # 如果得分小于 0.5，删除图片和 txt 文件
            if reward < 0.526:
                os.remove(img_path)
                os.remove(txt_path)
                print(f"图片 {img_path} 和其对应的文字 {txt_path} 已被删除")
        else:
            print(f"处理图片 {filename} 时出现错误: {result.get('error')}")
    
    except Exception as e:
        print(f"请求处理图片 {filename} 时出现异常: {str(e)}")

# 获取所有图片文件
image_files = [f for f in os.listdir(folder_path) if f.endswith((".jpg", ".png", ".jpeg", ".webp"))]

# 使用线程池并发处理
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    # 提交所有任务并行执行
    futures = {executor.submit(process_image, filename): filename for filename in image_files}

    # 处理完成的任务
    for future in as_completed(futures):
        filename = futures[future]
        try:
            future.result()  # 如果线程抛出异常，这里会捕获
        except Exception as e:
            print(f"处理图片 {filename} 时出现异常: {str(e)}")