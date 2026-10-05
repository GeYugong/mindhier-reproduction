---
date: 2026-10-05
timezone: Asia/Shanghai
status: running
---
# 环境与数据准备

服务器独立仓库已建立，初始提交为 71057f5。本地 gh 创建 GeYugong/mindhier-reproduction 私有仓库，并通过 HTTPS 凭据助手完成初次推送。提交身份沿用本地 Git 配置。

专用运行环境采用项目内 venv，继承只读基础环境中的 PyTorch/CUDA 等运行库，并在项目目录内安装独立依赖覆盖层。原环境未修改；来源及版本分别保存在 manifests/environment-base.json 和 manifests/environment-freeze.txt。

服务器直接连接 huggingface.co 超时，hf-mirror.com 可达。因此使用本地读取的官方 Hugging Face revision、文件长度与 SHA-256/Git blob ID 锁定下载清单，经镜像传输后校验内容。清单见 manifests/downloads.json，初期准备 Subject 1 数据、CLIP、Switti 和 VQ-VAE，共约 14.3 GB。

当前 GPU 上存在其他任务，不终止既有进程。训练卡数和每卡 microbatch 将依据试运行确定；对比损失 batch=512 的语义必须保持，不以简单累积替代全局负样本集合。
