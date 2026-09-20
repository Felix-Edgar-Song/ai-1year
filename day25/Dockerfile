FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn tiktoken transformers
RUN pip install --no-cache-dir vllm
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128
COPY app_vllm.py .
CMD ["python", "-m", "uvicorn", "app_vllm:app", "--host", "0.0.0.0", "--port", "8000"]
