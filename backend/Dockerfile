FROM python:3.10-slim
WORKDIR /app

COPY app.py /app/
COPY requirements.txt /app/
COPY entrypoint.sh /app/
COPY utils /app/utils
COPY models /app/models
COPY login_wandb.py /app/

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --upgrade pip

# Use torch 2.5.1+cpu for smaller image
RUN pip install --no-cache-dir \
    torch==2.5.1+cpu torchvision==0.20.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r /app/requirements.txt

RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]