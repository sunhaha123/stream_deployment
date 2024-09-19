from fastapi import FastAPI, File, UploadFile, Response
from fastapi.responses import StreamingResponse
from PIL import Image
import numpy as np
import torch
import io
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle
import asyncio
import time
import os
import cv2 
import time
import uvicorn
from contextlib import asynccontextmanager
import logging
from logging.handlers import RotatingFileHandler
from mmedit.apis import init_model, restoration_inference
from pydantic import BaseModel


app = FastAPI()
model = None
streamer = None

LOG_FMT = '[%(levelname)s] %(asctime)s: %(message)s'
LOG_DATE_FMT = '%Y-%m-%d %H:%M:%S'

formatter = logging.Formatter(fmt=LOG_FMT, datefmt=LOG_DATE_FMT)
if not os.path.exists('/opt/web/fastapi/log'):
    os.makedirs('/opt/web/fastapi/log')
rotatingHandler = RotatingFileHandler('/opt/web/fastapi/log/active.log', maxBytes=10000, backupCount=5)
rotatingHandler.setLevel(logging.INFO)
rotatingHandler.setFormatter(formatter)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)

logging.basicConfig(format=LOG_FMT, datefmt=LOG_DATE_FMT, level=logging.INFO, handlers=[rotatingHandler, console])


class RewardScoreRequest(BaseModel): 
    img_path: str
    # prompt: str


@app.on_event("startup")
async def startup_event():
    global model
    config = './main/configs/clipiqa_attribute_test.py'
    model = init_model(
        config, None, device=torch.device('cuda'))
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     global streamer
#     # 初始化 Streamer
#     streamer = Streamer(ManagedRemovetModel, batch_size=3, max_latency=0.01, worker_num=1, cuda_devices=(0,))
#     yield
#     # 清理资源
#     streamer = None



def open_image_with_opencv(contents: bytes):
    # 使用OpenCV读取图像
    image_array = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    # 将BGR格式转换为RGB格式
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 转换为Pillow格式
    return Image.fromarray(image_rgb)


@app.post("/compute-score")
async def compute_image_reward(request: RewardScoreRequest):
    img_path = request.img_path
    # prompt = request.prompt
    logging.info(f"Received request with img_path: {img_path} !")
    try:
        # 验证图像文件是否存在
        if not os.path.exists(img_path):
            return {"error": f"Image path '{img_path}' does not exist."}

        # 开始计时
        start_time = time.perf_counter()

        output, attributes = restoration_inference(model, img_path, return_attributes=True)
        output = output.float().detach().cpu().numpy()
        attributes = attributes.float().detach().cpu().numpy()[0]
       
        print(attributes)

        # 记录模型处理时间
        end_time = time.perf_counter()
        logging.info(f"Total processing time: {end_time - start_time:.6f} seconds")

        # 返回模型的评分结果
        return {"score": str(attributes[0])}

    except Exception as e:
        logging.error(f"Error while processing image: {e}")
        return {"error": str(e)}


@app.get("/status")
async def status():
    return Response(status_code=200)


if __name__ == "__main__":
      uvicorn.run('api:app', host='0.0.0.0', port=8080, reload=False, workers=1)
      