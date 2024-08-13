from fastapi import FastAPI, File, UploadFile, Response
from fastapi.responses import StreamingResponse
from PIL import Image
import numpy as np
import torch
from transparent_background import Remover
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


from service_streamer.service_streamer import ManagedModel
from remove_background.remover import Remover
from service_streamer.service_streamer import Streamer

app = FastAPI()
model = None
streamer = None

class ManagedRemovetModel(ManagedModel):
   
    def init_model(self):
        self.model = Remover()

    def predict(self, img):
        return self.model.process_batch(img, type='rgba')


@app.on_event("startup")
async def startup_event():
    global streamer
    streamer = Streamer(ManagedRemovetModel, batch_size=1, max_latency=0.01, worker_num=1, cuda_devices=(0,))
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     global streamer
#     # 初始化 Streamer
#     streamer = Streamer(ManagedRemovetModel, batch_size=3, max_latency=0.01, worker_num=1, cuda_devices=(0,))
#     yield
#     # 清理资源
#     streamer = None
def setup_logging():
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


def open_image_with_opencv(contents: bytes):
    # 使用OpenCV读取图像
    image_array = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    # 将BGR格式转换为RGB格式
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 转换为Pillow格式
    return Image.fromarray(image_rgb)


@app.post("/remove-background")
async def remove_background(file: UploadFile = File(...)):
    start_time = time.perf_counter()

    # 读取上传的图像文件
    t0 = time.time()
    contents = await file.read()
    logging.info("read wasted %5f"%(time.time()-t0))

    t0 = time.time()
    img = open_image_with_opencv(contents)        
    logging.info("open wasted %5f"%(time.time()-t0))

    t0 = time.time()
    mid = streamer.predict([img])[0]
    logging.info("steamer wasted %5f"%(time.time()-t0))

    t0 = time.time()    
    mid_array = np.array(mid)    
    if mid_array.shape[2] == 4:  # 检查是否为 RGBA
        mid_array = cv2.cvtColor(mid_array, cv2.COLOR_RGBA2BGRA)
    elif mid_array.shape[2] == 3:  # 如果是 RGB
        mid_array = cv2.cvtColor(mid_array, cv2.COLOR_RGB2BGR)
    # 使用 OpenCV 将图像编码为 PNG 格式
    _, buffer = cv2.imencode('.png', mid_array, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    # 将编码后的图像数据转换为字节数组
    img_byte_arr = io.BytesIO(buffer)    
    logging.info("save wasted %5f"%(time.time()-t0))

    end_time = time.perf_counter()
    logging.info(f"Total processing time: {end_time - start_time:.6f} seconds")
    # 返回流响应
    return StreamingResponse(img_byte_arr, media_type="image/png", headers={"Content-Disposition": "inline; filename=remove.png"})


@app.get("/status")
async def status():
    return Response(status_code=200)


if __name__ == "__main__":
      uvicorn.run('api:app', host='0.0.0.0', port=8080, reload=False, workers=3)