# Baizhu-LLM 白猪大模型 Termux部署指南
安卓Termux（安卓Linux容器）运行白猪大模型

## 环境要求
- ARM64(aarch64)安卓手机
- 可用内存 ≥2GB，推荐3GB+
- Python3.10+

## 安装步骤
```bash
pkg update && pkg upgrade -y
pkg install python git
git clone https://github.com/ynno917/baizhu-llm.git
cd baizhu-llm
pip install -r requirements.txt
python src/inference.py

