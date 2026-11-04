FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/
COPY data/ data/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[train]"

EXPOSE 5000

ENTRYPOINT ["python3"]
CMD ["scripts/train.py"]

