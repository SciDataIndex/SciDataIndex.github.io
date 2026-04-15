#FROM python:3.12-slim
FROM acr-openxlab-prod-registry-vpc.cn-shanghai.cr.aliyuncs.com/public/python:3.12-slim
WORKDIR /app

COPY . .

EXPOSE 8000

CMD ["python3", "-m", "http.server", "8000"]
