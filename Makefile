# Makefile for Medical AI Knowledge Base
.PHONY: install install-dev test lint format clean run docker-build docker-run help

# 默认目标
help:
	@echo "Medical AI Knowledge Base - 开发命令"
	@echo "======================================"
	@echo "install      - 安装生产依赖"
	@echo "install-dev  - 安装开发依赖"
	@echo "test         - 运行测试"
	@echo "test-cov     - 运行测试并生成覆盖率报告"
	@echo "lint         - 运行代码检查"
	@echo "format       - 格式化代码"
	@echo "clean        - 清理缓存和临时文件"
	@echo "run          - 启动应用"
	@echo "docker-build - 构建Docker镜像"
	@echo "docker-run   - 运行Docker容器"
	@echo "help         - 显示此帮助信息"

# 安装
install:
	pip install -r requirements.txt

install-dev: install
	pip install -r requirements-dev.txt

# 测试
test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

# 代码质量
lint:
	flake8 src/ tests/ --max-line-length=100 --exclude=__pycache__
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/
	isort src/ tests/

# 清理
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

# 运行
run:
	streamlit run app.py

# Docker
docker-build:
	docker build -t medical-ai-assistant .

docker-run:
	docker run -p 8501:8501 -v $$(pwd)/medical_db:/app/medical_db medical-ai-assistant

# 预提交钩子
setup-hooks:
	pre-commit install

# 环境变量检查
check-env:
	@echo "Checking environment variables..."
	@python -c "import os; required = ['OLLAMA_HOST']; [print(f'{var}: {os.getenv(var, \"NOT SET\")}') for var in required]"