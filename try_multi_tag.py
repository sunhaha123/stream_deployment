from concurrent.futures import ThreadPoolExecutor
import onnxruntime as ort
from PIL import Image
import numpy as np
import os
import csv
import time

# 默认配置
defaults = {
    "model": "wd-v1-4-moat-tagger-v2",
    "threshold": 0.35,
    "character_threshold": 0.85,
    "replace_underscore": False,
    "trailing_comma": False,
    "exclude_tags": "",
    "ortProviders": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "HF_ENDPOINT": "https://huggingface.co"
}

# 模型路径
models_dir = '/mnt/model/wd-swinv2-tagger-v3'
model_path = '/mnt/model/wd-swinv2-tagger-v3/model.onnx'

# 全局变量用来存储模型和标签
model = None
tags = []
general_index = None
character_index = None

# 从CSV文件中读取标签并确定不同类别的索引
def load_tags(models_dir, replace_underscore=True):
    global tags, general_index, character_index
    tags = []
    general_index = None
    character_index = None
    with open(os.path.join(models_dir, "selected_tags.csv")) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if general_index is None and row[2] == "0":
                general_index = reader.line_num - 2
            elif character_index is None and row[2] == "4":
                character_index = reader.line_num - 2
            if replace_underscore:
                tags.append(row[1].replace("_", " "))
            else:
                tags.append(row[1])

# 初始化模型和标签（只加载一次）
def initialize_model():
    global model
    model = ort.InferenceSession(model_path, providers=defaults["ortProviders"])
    load_tags(models_dir)

# 定义一个函数来处理单张图片并保存结果
def process_image(image_path):
    global model, tags, general_index, character_index

    # 获取输入形状
    input = model.get_inputs()[0]
    height = input.shape[1]

    # 加载图片
    image = Image.open(image_path)

    # 调整图片尺寸并填充为正方形
    ratio = float(height) / max(image.size)
    new_size = tuple([int(x * ratio) for x in image.size])
    image = image.resize(new_size, Image.LANCZOS)
    square = Image.new("RGB", (height, height), (255, 255, 255))
    square.paste(image, ((height - new_size[0]) // 2, (height - new_size[1]) // 2))

    # 将图像转换为模型输入格式
    image = np.array(square).astype(np.float32)
    image = image[:, :, ::-1]  # RGB -> BGR
    image = np.expand_dims(image, 0)

    # 执行推理，获取模型输出
    label_name = model.get_outputs()[0].name
    probs = model.run([label_name], {input.name: image})[0]

    result = list(zip(tags, probs[0]))

    # 获取general和character类别的标签
    general = [item for item in result[general_index:character_index] if item[1] > defaults['threshold']]
    character = [item for item in result[character_index:] if item[1] > defaults['character_threshold']]

    # 合并标签并进行过滤
    all_tags = character + general
    remove = [s.strip() for s in defaults["exclude_tags"].lower().split(",")]
    all_tags = [tag for tag in all_tags if tag[0] not in remove]

    # 生成最终的标签字符串
    res = ("" if defaults["trailing_comma"] else ", ").join(
        (item[0].replace("(", "\\(").replace(")", "\\)") + (", " if defaults["trailing_comma"] else "") for item in all_tags)
    )

    # 获取图片文件名
    image_filename = os.path.splitext(os.path.basename(image_path))[0]
    output_txt_path = os.path.join(os.path.dirname(image_path), f"{image_filename}.txt")

    # 将结果保存到txt文件
    with open(output_txt_path, 'w') as txt_file:
        txt_file.write(res)

    print(f"Processed {image_filename}, results saved to {output_txt_path}")


# 遍历文件夹下的所有图片并处理
def process_images_in_folder(folder_path):
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')  # 支持的图片格式
    image_paths = []
    
    # 遍历文件夹，获取所有图片路径
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(supported_formats):
                image_paths.append(os.path.join(root, file))
    
    # 使用线程池进行并行处理
    num_threads = min(12, len(image_paths))  # 使用最大可用CPU核心数
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        executor.map(process_image, image_paths)


# 指定要处理的文件夹路径
image_folder_path = '/home/echo/workspace/stream_deployment/tusi-testing'

# 处理文件夹中的所有图片
if __name__ == '__main__':
    t0 = time.time()
    initialize_model()  # 模型和标签仅初始化一次
    process_images_in_folder(image_folder_path)
    print(f"Total processing time: {time.time() - t0:.2f} seconds")