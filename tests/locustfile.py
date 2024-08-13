from locust import HttpUser, task, between, TaskSet
import os

class UserBehavior(TaskSet):
    def on_start(self):
        # 在任务开始时读取文件到内存
        with open("test_jpeg.jpg", "rb") as image_file:
            self.image_data = image_file.read()

    @task
    def remove_background(self):
        # 使用预先读取的文件数据进行请求
        self.client.post("/remove-background", files={"file": ("test_image.png", self.image_data)})

class WebsiteUser(HttpUser):
    tasks = [UserBehavior]
    wait_time = between(0.5, 1)  # 模拟用户在请求之间等待0.5到1秒