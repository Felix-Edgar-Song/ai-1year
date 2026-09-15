from locust import HttpUser, task
class FastAPIUser(HttpUser):
    @task
    def gen(self):
        self.client.get("/generate?prompt=EN:%20Hello%20KO:")
