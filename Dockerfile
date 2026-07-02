FROM python:3.10-slim

WORKDIR /app

# 系统依赖（PDF 解析需要 poppler）
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖（先复制 requirements 以利用 Docker 缓存层）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制项目代码
COPY . .

# 创建日志目录
RUN mkdir -p logs

EXPOSE 8000

CMD ["python", "-m", "src.api.main"]
