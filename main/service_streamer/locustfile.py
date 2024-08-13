from locust import HttpUser, TaskSet, task, between

class UserBehavior(TaskSet):
    @task
    def remove_background(self):
        # 假设你有一个名为 'test_image.png' 的测试图像文件在当前目录
        with open("12.png", "rb") as image_file:
            self.client.post("/stream", files={"file": image_file})

class WebsiteUser(HttpUser):
    tasks = [UserBehavior]
    wait_time = between(1, 3)  # 模拟用户在请求之间等待1到3秒
    