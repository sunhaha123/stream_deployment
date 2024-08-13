from flask import Flask, request, send_file
from flask import Response
from PIL import Image
import os
import io
import time
import cv2
import numpy as np
#框架中已经包含了多线程队列，因此不要再使用gevent
# from gevent.pywsgi import WSGIServer
from service_streamer.service_streamer import ManagedModel
from remove_background.remover import Remover
from service_streamer.service_streamer import Streamer


app = Flask(__name__)
model = None
streamer = None



class ManagedRemovetModel(ManagedModel):
   
    def init_model(self):
        self.model = Remover()

    def predict(self, img):
        return self.model.process_batch(img, type='rgba')



@app.route('/remove-background', methods=['POST'])
def predict():
    if request.method == 'POST':
        t0 = time.time()
        file = request.files['file']
        img_bytes = file.read()
        print("read wasted %5f"%(time.time()-t0))
        t0 = time.time()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        print("open wasted %5f"%(time.time()-t0))
        t0 = time.time()
        mid = streamer.predict([img])[0]
        print("steamer wasted %5f"%(time.time()-t0))
        t0 = time.time()
        # 将 PIL.Image 对象转换为 NumPy 数组
        mid_array = np.array(mid)
        # 如果图像有 alpha 通道（即 RGBA 格式），可能需要转换为 BGRA 格式
        # OpenCV 默认使用 BGR 排列
        if mid_array.shape[2] == 4:  # 检查是否为 RGBA
            mid_array = cv2.cvtColor(mid_array, cv2.COLOR_RGBA2BGRA)
        elif mid_array.shape[2] == 3:  # 如果是 RGB
            mid_array = cv2.cvtColor(mid_array, cv2.COLOR_RGB2BGR)
        # 使用 OpenCV 将图像编码为 PNG 格式
        _, buffer = cv2.imencode('.png', mid_array, [cv2.IMWRITE_PNG_COMPRESSION, 4])
        # 将编码后的图像数据转换为字节数组
        img_byte_arr = io.BytesIO(buffer)
        print("save wasted %5f"%(time.time()-t0))
        return send_file(img_byte_arr, mimetype='image/png', as_attachment=False, download_name='remove.png')


@app.route('/t')
def hello():
    return 'Hello World'


@app.route("/healthz")
def health():
    return Response(status=200)


@app.route("/status")
def status():
    return Response(status=200)



if __name__ == "__main__":
    streamer = Streamer(ManagedRemovetModel, batch_size=3, max_latency=0.01, worker_num=1, cuda_devices=(0,))
    app.run(host="0.0.0.0", port=8080, )



    #通过图片测试
    # with open(r"cat.jpg", 'rb') as f:
    #     image_bytes = f.read()
    # with open(r"truck.jpg", 'rb') as f:
    #     image_bytes2 = f.read()
    # outputs = streamer.predict([image_bytes,image_bytes2])
    # print(outputs)
