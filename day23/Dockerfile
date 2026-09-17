FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime
WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn tiktoken
COPY app_v2.py .
CMD ["python", "-m", "uvicorn", "app_v2:app", "--host", "0.0.0.0", "--port", "8000"]
