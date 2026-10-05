# MindHier 复现

项目本地目录：`D:\0code\Research\x_neuroadapter\mindhier-reproduction`。

2026-10-05：当前目标为双 RTX 5090，连接、BF16 反向和双卡 NCCL 已验证。服务器独立仓库已建立在个人根目录下：`/data/matengyu/geyugong/mindhier-reproduction/`。采用服务器提交、本地同步、本地 gh/Git 推送 GitHub 私有仓库的工作流程。完整计划见 [双 RTX 5090 复现计划](PLAN_5090_ZH.md)。当前已启动 batch=512 的真实模型显存与恢复检查；完整缓存、100 次试运行及正式阶段一训练由 scripts/run_subject1.py 顺序推进。私有仓库：https://github.com/GeYugong/mindhier-reproduction 。

运行配置及差异见 [实施记录](protocols/IMPLEMENTATION_NOTES.md)，实时状态由 scripts/status.py 读取。每小时检查已启用；尚未产生正式解码结果或八指标。

以下保留 2026-10-03（北京时间）的初次环境检查、论文阅读和目录整理记录。原目录名为“新建文件夹”；重命名前后两份 PDF 的 SHA-256 校验一致。该次检查尚未安装专用环境、下载大规模数据或启动训练。

## 论文与代码

- 英文原文：`2510.22335v2.pdf`，26 页；本文记录以英文原文为依据。
- 现有中文译本：`MindHier_Chinese_Translation.pdf`。
- 标题：Moving Beyond Diffusion: Hierarchy-to-Hierarchy Autoregression for fMRI-to-Image Reconstruction。
- 作者：Xu Zhang、Ruijie Quan、Wenguan Wang、Yi Yang；ICLR 2026。
- [arXiv v2](https://arxiv.org/abs/2510.22335v2)，2026-06-10 修订。
- [官方代码](https://github.com/XuZhang2/MindHier)已克隆至 `upstream/`，未修改。
- 检查的代码提交：`dfcdfdcb60acad9f8e9a88785e5b2776f828f669`（2026-06-27）。

## 这篇论文讲什么

任务是根据受试者看图片时的 fMRI 脑活动重建视觉刺激。核心主张是：脑信号应提供多层信息，并在图像生成的不同尺度发挥不同作用。

作者指出，许多扩散式重建方法将脑信号压缩成单一表示，在整个生成过程反复使用；这与“先确定大致内容，再补充局部细节”的生成需求存在不匹配。MindHier 将分层脑特征与从粗到细的图像生成相结合（原文 §1、§3，pp.1–6）。

1. **分层 fMRI 编码器**：从一份 fMRI 输入中提取多个层级的表示。较深层侧重语义，较浅层保留相对更细的视觉信息。这是学习到的层级，不是直接把输入按脑区划成几组（§3.1，pp.3–4；附录 B、D）。
2. **层级对齐**：让这些表示分别对齐冻结的 CLIP ViT-L/14 的第 12、16、20、24 层。各层用归一化后的 MSE 对齐，最后一层还用图像和文字的 SoftCLIP 目标约束语义（§3.2，pp.4–5；§4.4/Table 3，pp.8–9）。
3. **按尺度引导生成**：基于 Switti 的尺度自回归生成器先使用深层语义特征建立低分辨率整体，再通过带掩码的交叉注意力引入较浅层特征，逐步补充结构和纹理。经过 10 个尺度，VQ-VAE 解码为 512×512 图像（§3.3，pp.5–6；附录 E，p.19）。

可以将其理解为：先根据脑信号决定“画面大致是什么”，再逐步决定“各个部分怎么画”。这里的自回归单位是下一尺度的二维 token 图，而非逐像素扫描。

## 结果如何解读

主实验采用 NSD 新版 1,000 张测试图，测试时对重复 fMRI 响应取平均；训练使用非平均的单次试验。主表对 Subject 1、2、5、7 取平均（§4.1，p.6）。

| 原文 Table 1 指标 | MindHier | MindEye2 |
| --- | ---: | ---: |
| CLIP 二选一识别率 ↑ | 96.4% | 93.0% |
| SwAV 距离 ↓ | 0.329 | 0.344 |
| 像素相关性 ↑ | 0.235 | 0.322 |
| SSIM ↑ | 0.381 | 0.431 |
| 每张图推理耗时 ↓ | 2.64 秒 | 12.14 秒 |

这是原文表格中的比较，MindEye2 一行使用低层辅助特征，而此处 MindHier 为不带辅助特征的版本；不应将其解释为所有设置完全相同或所有指标均领先。

- 强项是语义保真和速度；CLIP 96.4% 是二选一识别指标，不是“96.4% 的像素还原正确”。
- 原文另报告 MindHier†（带低层辅助特征）：PixCorr 0.326、SSIM 0.461、CLIP 95.4%；应作为单独实验设置。
- 主结果使用 4 个候选并按 CLIP 相似度选择；附录 Table S5（p.17）报告单候选 0.92 秒、CLIP 95.0%，4 候选 2.64 秒、CLIP 96.4%。上述耗时是论文测量，尚未在本项目复测。
- 所有推理实验在单张 RTX 4090 上完成（附录 E，p.19）。原文没有在该处明确训练 GPU 数量或训练显存峰值。
- 文中“更稳定”不等于没有随机采样；细节、文字、面孔和物体数量仍可能出错（附录 G、H，pp.20–21）。
- 正文声称较 MindEye2 快 4.67 倍，但按 Table 1 的 12.14/2.64 计算约为 4.60 倍；复现应直接记录实测耗时与采样设置。

## 历史环境实测：双 RTX 4090（2026-10-03）

当次检查对象为双 RTX 4090 服务器；与当前目标双 RTX 5090 及本地 Windows 均为不同环境。

| 项目 | 2026-10-03 检查结果 |
| --- | --- |
| SSH 别名 | `neuroadapter-4090`，与 `双卡4090` 指向同一主机 |
| 登录目标 | `tyma@10.8.176.61:10006` |
| 操作系统 | Ubuntu 22.04.5 LTS |
| CPU | Intel Core i9-14900KF，32 个逻辑 CPU |
| 内存 | 125 GiB 总量，检查时约 121 GiB 可用 |
| GPU | 2×NVIDIA GeForce RTX 4090，每张 24,564 MiB |
| 初始占用 | GPU 0：54 MiB；GPU 1：15 MiB；利用率均 0%，无计算进程 |
| 驱动 | 580.178.04 |
| GPU 拓扑 | PHB，经 PCIe 主桥连接 |
| 系统盘 | 约 70 GiB 可用，使用率 96% |
| `/data1` | 约 108 GiB 可用，使用率 99% |
| 系统 `nvcc` | CUDA Toolkit 11.5 |

使用现有 `/home/tyma/miniconda3/envs/smarterneuroadapter/bin/python` 做只读环境探测和小规模计算验证：

- Python 3.10.19；PyTorch 2.9.1；torchvision 0.24.1；PyTorch CUDA 运行时 12.8。
- 两张 GPU 均可见；每张 GPU 的实际矩阵乘法通过；均支持 BF16。
- 双进程 NCCL `all_reduce` 通过：两个 rank 分别提供 1、2，都得到 3。
- 现有包：transformers 5.12.1、accelerate 1.14.0、diffusers 0.38.0、h5py 3.16.0、safetensors 0.8.0、NumPy 2.2.6。
- 该环境缺少 webdataset；官方 requirements 要求 NumPy <2.0，现有环境不满足。其余依赖尚未逐项验证。
- 系统 Toolkit 11.5 与已测 PyTorch 的 CUDA 12.8 运行时属于不同组件；GPU 计算已通过，若后续编译 CUDA 扩展，还需单独核对编译工具链。

结论：服务器连接、GPU 计算与双卡通信可用；MindHier 专用依赖、数据和训练显存尚未验证，不能将其表述为训练已就绪。本轮未修改共享环境或其他项目文件。

## 已找到的数据与缺口

只检查了 `/data1/matengyu/geyugong/` 中与已有脑重建项目相关的数据，并非全盘盘点。

- NSD 图像：`/data1/matengyu/geyugong/neuroadapter-subject1-research/data/raw/nsd/stimuli/nsd_stimuli.hdf5`。
  - 只读验证：`imgBrick`，形状 `(73000, 425, 425, 3)`，uint8，约 39.56 GB。
- Subject 1 表面脑数据：`/data1/matengyu/geyugong/neuroadapter-subject1-research/data/derived/neural_data/betas_sub-01.h5`。
  - 只读验证：`lh_betas`、`rh_betas`，分别为 `(30000, 163842)`，float32。

已有图像有望复用，但必须核对 NSD ID、图像预处理及训练/测试划分。已有表面脑数据不能直接替代本文的体积空间输入：原文附录 B（p.15）采用 1.8-mm native volume + nsdgeneral mask；官方代码对 Subject 1 期望 15,724 个体素。

官方加载器需要 `betas_all_subj01_fp32_renorm.hdf5` 的 `betas` 数据集、`wds/subj01/train` 和 `new_test`、图像目录和 COCO caption 缓存。此轮尚未确认这些材料已按要求就绪，也未确认 Switti/VQ-VAE 所需权重就绪。

## 后续复现应遵循的配置

来源：原文附录 E（p.19），而非直接采用仓库默认参数。

| 配置 | 阶段一：分层脑编码器 | 阶段二：尺度生成器 |
| --- | --- | --- |
| 训练对象 | fMRI 编码器；CLIP 冻结 | 生成器微调；fMRI 编码器冻结 |
| 优化器 | AdamW，β=(0.9,0.999) | AdamW，β=(0.9,0.95) |
| 学习率 | 1e-4，warmup + cosine | 1e-4，warmup + cosine |
| 论文 batch size | 512 | 80 |
| 训练长度 | 300 epochs | 9,000 iterations |
| 其他 | MSE 权重 2e5，SoftCLIP 温度 0.005 | 30 层，512×512，10 个尺度，Switti 预训练 |

已发现的源码与复现注意项：

1. `configs/default.yaml` 仅是参考文档，不会被训练入口读取；其部分数值与论文不同。实际入口为 `brainencoder/train.py` 和 `MindHier/train.py`。
2. 阶段一脚本默认 batch=64、epochs=100；阶段二默认 bs=768、max_iters=1,000,000。均不能直接当成论文配置。
3. 阶段二只有显式指定 `hierarchy_ar_ckpt` 才加载生成器权重。省略时不是论文描述的 Switti 预训练微调路径；必须核对权重来源和成功加载的参数。
4. 阶段二已有 FSDP、梯度检查点和梯度累积相关实现，但仍需真实前向/反向显存测试。当前累积路径在 `trainer.py` 和 `utils/amp_sc.py` 中未将 loss 除以累积次数；使用累积前应验证并处理有效 batch、梯度尺度与学习率语义。
5. 阶段一包含批内对比损失，简单用小批量梯度累积并不等价于论文的大批量负样本集合；需明确记录适配方案及偏差。
6. 48 GB 总显存分布在两张各 24 GB 的 GPU 上。单卡推理有论文依据；双卡完整训练能否保持论文设置，需要模型与优化器分片、混合精度及小批量测试后确认。

建议执行顺序：锁定数据格式与新版测试划分 → 创建项目专用环境并固定依赖 → 核对 CLIP/Switti/VQ-VAE 权重 → Subject 1 小规模前向与反向验证 → 阶段一/阶段二训练 → 按论文的单候选与四候选协议评估 → 扩展至 Subject 2、5、7。主表复现与 Subject 1 单独复现应分别报告。
