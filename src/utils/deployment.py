"""
部署和运维工具
提供Docker、Kubernetes、监控等基础设施
"""
from typing import Dict, Any, List
import logging
import subprocess
import os

logger = logging.getLogger(__name__)

class DockerConfigurator:
    """Docker配置生成器"""
    
    def __init__(self, project_name: str = "medical-ai"):
        """初始化配置生成器"""
        self.project_name = project_name
    
    def generate_dockerfile(self) -> str:
        """生成Dockerfile"""
        return f"""# 医学AI助手 - Docker镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \\
    git \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 创建日志目录
RUN mkdir -p /app/logs

# 设置环境变量
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# 暴露端口
EXPOSE 8501

# 健康检查
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# 启动应用
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
"""
    
    def generate_docker_compose(self) -> str:
        """生成Docker Compose配置"""
        return f"""version: '3.8'

services:
  # 主应用
  app:
    build: .
    container_name: {self.project_name}-app
    ports:
      - "8501:8501"
    volumes:
      - ./medical_db:/app/medical_db
      - ./logs:/app/logs
      - ./.env:/app/.env
    environment:
      - OLLAMA_HOST=ollama
      - OLLAMA_PORT=11434
      - LOG_LEVEL=INFO
    depends_on:
      - ollama
      - redis
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 3s
      retries: 3
      start_period: 5s

  # Ollama服务
  ollama:
    image: ollama/ollama:latest
    container_name: {self.project_name}-ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    environment:
      - OLLAMA_HOST=0.0.0.0
      - OLLAMA_PORT=11434
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
      interval: 30s
      timeout: 3s
      retries: 3
      start_period: 10s

  # Redis缓存
  redis:
    image: redis:7-alpine
    container_name: {self.project_name}-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 3

  # Prometheus监控
  prometheus:
    image: prom/prometheus:latest
    container_name: {self.project_name}-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    restart: unless-stopped

  # Grafana可视化
  grafana:
    image: grafana/grafana:latest
    container_name: {self.project_name}-grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    restart: unless-stopped

volumes:
  ollama_data:
  redis_data:
  prometheus_data:
  grafana_data:
"""
    
    def generate_prometheus_config(self) -> str:
        """生成Prometheus配置"""
        return """global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'medical-ai'
    static_configs:
      - targets: ['app:8501']
    metrics_path: '/metrics'
    scrape_interval: 10s
  
  - job_name: 'ollama'
    static_configs:
      - targets: ['ollama:11434']
    scrape_interval: 30s
  
  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
    scrape_interval: 30s
"""

class KubernetesConfigurator:
    """Kubernetes配置生成器"""
    
    def __init__(self, app_name: str = "medical-ai"):
        """初始化K8s配置器"""
        self.app_name = app_name
    
    def generate_deployment(self) -> str:
        """生成Deployment配置"""
        return f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {self.app_name}
  namespace: medical-ai
spec:
  replicas: 3
  selector:
    matchLabels:
      app: {self.app_name}
  template:
    metadata:
      labels:
        app: {self.app_name}
    spec:
      containers:
      - name: {self.app_name}
        image: medical-ai:latest
        ports:
        - containerPort: 8501
        env:
        - name: OLLAMA_HOST
          value: "ollama-service"
        - name: REDIS_HOST
          value: "redis-service"
        - name: LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /_stcore/health
            port: 8501
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /_stcore/health
            port: 8501
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: medical-db
          mountPath: /app/medical_db
      volumes:
      - name: medical-db
        persistentVolumeClaim:
          claimName: medical-db-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: {self.app_name}-service
  namespace: medical-ai
spec:
  selector:
    app: {self.app_name}
  ports:
  - protocol: TCP
    port: 8501
    targetPort: 8501
  type: LoadBalancer
"""
    
    def generate_ingress(self, domain: str = "medical-ai.example.com") -> str:
        """生成Ingress配置"""
        return f"""apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {self.app_name}-ingress
  namespace: medical-ai
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - {domain}
    secretName: {self.app_name}-tls
  rules:
  - host: {domain}
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: {self.app_name}-service
            port:
              number: 8501
"""

class CIConfigurator:
    """CI/CD配置生成器"""
    
    def generate_github_actions(self) -> str:
        """生成GitHub Actions配置"""
        return """name: Medical AI CI/CD

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  PYTHON_VERSION: "3.11"

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run linting
      run: |
        flake8 src/ --max-line-length=100
        mypy src/ --ignore-missing-imports
    
    - name: Run tests
      run: |
        pytest tests/ -v --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
  
  build:
    needs: test
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Login to DockerHub
      uses: docker/login-action@v2
      with:
        username: ${{ secrets.DOCKERHUB_USERNAME }}
        password: ${{ secrets.DOCKERHUB_TOKEN }}
    
    - name: Build and push
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: |
          medical-ai:latest
          medical-ai:${{ github.sha }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
  
  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to Kubernetes
      uses: azure/k8s-deploy@v4
      with:
        manifests: |
          k8s/deployment.yaml
          k8s/ingress.yaml
        kubeconfig: ${{ secrets.KUBE_CONFIG }}
"""

class MonitoringSetup:
    """监控设置"""
    
    def generate_grafana_dashboard(self) -> str:
        """生成Grafana仪表盘配置"""
        return """{
  "dashboard": {
    "title": "Medical AI Monitor",
    "panels": [
      {
        "title": "Query Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(query_duration_seconds_bucket[5m]))"
          }
        ]
      },
      {
        "title": "Cache Hit Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "cache_hits / (cache_hits + cache_misses) * 100"
          }
        ]
      },
      {
        "title": "Active Users",
        "type": "stat",
        "targets": [
          {
            "expr": "active_sessions"
          }
        ]
      }
    ]
  }
}"""

# 使用示例
if __name__ == "__main__":
    # 1. Docker配置
    docker_config = DockerConfigurator()
    
    with open("Dockerfile", "w") as f:
        f.write(docker_config.generate_dockerfile())
    
    with open("docker-compose.yml", "w") as f:
        f.write(docker_config.generate_docker_compose())
    
    with open("prometheus.yml", "w") as f:
        f.write(docker_config.generate_prometheus_config())
    
    # 2. Kubernetes配置
    k8s_config = KubernetesConfigurator()
    
    os.makedirs("k8s", exist_ok=True)
    
    with open("k8s/deployment.yaml", "w") as f:
        f.write(k8s_config.generate_deployment())
    
    with open("k8s/ingress.yaml", "w") as f:
        f.write(k8s_config.generate_ingress())
    
    # 3. CI/CD配置
    os.makedirs(".github/workflows", exist_ok=True)
    
    ci_config = CIConfigurator()
    with open(".github/workflows/ci.yml", "w") as f:
        f.write(ci_config.generate_github_actions())
    
    # 4. 监控配置
    monitoring = MonitoringSetup()
    with open("grafana-dashboard.json", "w") as f:
        f.write(monitoring.generate_grafana_dashboard())
    
    print("✅ 所有配置文件生成成功！")
