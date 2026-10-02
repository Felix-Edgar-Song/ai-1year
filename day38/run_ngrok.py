from pyngrok import ngrok
import uvicorn, threading, time
def run_api():
    uvicorn.run("app_demo:app", host="0.0.0.0", port=8000, log_level="error")
threading.Thread(target=run_api, daemon=True).start()
time.sleep(3)
url = ngrok.connect(8000)
print(f"LIVE DEMO URL: {url}/vram")
print(f"LIVE DEMO URL: {url}/health")
print(f"LIVE DEMO URL: {url}/generate?prompt=Hello%20Meta%20SG")
input("Enter 눌러서 종료 - 이 URL을 README에 넣기!")
