FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn torch tiktoken
COPY app_v2.py /app/app_v2.py
CMD ["python", "-m", "uvicorn", "app_v2:app", "--host", "0.0.0.0", "--port", "8000"]
