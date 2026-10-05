---
title: MindHier 双 RTX 5090 复现计划
date: 2026-10-05
timezone: Asia/Shanghai
status: planned
paper: arXiv:2510.22335v2
upstream_commit: dfcdfdcb60acad9f8e9a88785e5b2776f828f669
---

# MindHier 双 RTX 5090 复现计划

本实验先完成 Subject 1 的端到端复现，再按相同协议完成 Subject 2、5、7，计算四受试者平均结果。主要目标是论文不带低层辅助特征的 MindHier，完整保留解码结果并计算 PixCorr、SSIM、AN2、AN5、Incep、CLIP、Eff、SwAV。主实验完成后再进行核心消融；带辅助特征的 MindHier† 单独建实验记录。

截至 2026-10-05，已完成连接、资源、部分数据和源码的只读预检。本文件为待执行计划，尚未建立新的服务器 Git 仓库、创建 GitHub 私有仓库、安装专用环境或启动训练。现有 `upstream/` 是作者仓库，本地复现目录本身尚无 Git 仓库。

## 1. 工作目录与推进方式

| 位置 | 路径或目标 | 用途 |
| --- | --- | --- |
| 服务器个人根目录 | `/data/matengyu/geyugong/` | 实测存在；Linux 路径为小写 |
| 新服务器仓库 | `/data/matengyu/geyugong/mindhier-reproduction/` | 在个人根目录直接新建，与其他项目并列 |
| 本地仓库 | `D:\0code\Research\x_neuroadapter\mindhier-reproduction` | 同步服务器提交、备份结果、使用本地 gh 上传 |
| GitHub | 拟建 `GeYugong/mindhier-reproduction`，private | 保存代码、协议、实验日志、指标及产物清单 |

服务器仓库作为执行与提交的主仓库。固定流程为：**服务器修改、验证、提交 → 本地拉取同一提交 → 本地 gh 建立或核验私有仓库、本地 git push → 核验三端提交一致**。本轮形成的计划及预检记录在初始化阶段导入服务器并作首次文档提交，此后正式变更在服务器完成。

新环境、缓存、处理后数据和运行结果均写入新项目目录。已有项目的数据只读复用，不修改原始文件。保留现有本地 PDF 和作者仓库，不覆盖其他项目。

## 2. 论文基线

依据英文原文 §3–4、Table 1–4、附录 B/C/E，以及固定版本的作者代码。论文在线版本信息已核验：[arXiv v2](https://arxiv.org/abs/2510.22335v2)；代码来源：[XuZhang2/MindHier](https://github.com/XuZhang2/MindHier)。

核心流程包括：分层 fMRI 编码器提取四层表示；对齐冻结 CLIP ViT-L/14 的第 12、16、20、24 层；以深层语义初始化生成，再用分尺度交叉注意力引入较浅层表示；经 10 个尺度和 VQ-VAE 解码得到 512×512 图像。阶段二基于 Switti 预训练生成器进行微调。

| 配置 | 阶段一：分层脑编码器 | 阶段二：自回归生成器 |
| --- | --- | --- |
| 训练对象 | fMRI 编码器；CLIP 冻结 | 生成器；fMRI 编码器冻结 |
| 优化器 | AdamW，β=(0.9, 0.999) | AdamW，β=(0.9, 0.95) |
| 学习率 | 1e-4，warmup + cosine | 1e-4，warmup + cosine |
| 论文 batch | 512 | 80 |
| 训练长度 | 300 epochs | 9,000 次优化器更新 |
| 核心设置 | L2 归一化 MSE，权重 2e5；SoftCLIP 温度 0.005 | 30 层、10 个尺度、512×512 随机裁剪 |
| 初始化 | 固定 CLIP；脑编码器按作者实现初始化 | 明确加载并验证 Switti 和 VQ-VAE 权重 |

来源：英文原文 p.19，附录 E。warmup 长度、部分正则化设置及训练 GPU 数量在该处没有明确给出，实施时按固定源码与训练预检形成配置，并在偏差表中注明出处，不能写成论文已明确报告的参数。

数据使用 1.8-mm native volume 中经 nsdgeneral mask 筛选的体素，训练保留单次 trial，测试按图像 ID 汇总重复响应。正式测试使用新版 shared 1,000 图像协议，不能与旧版 982 张协议混合。实际训练/测试 trial 数从 manifest 核验，不将正文约数直接写死为采样数。

## 3. 预检结论与实施前必须解决的问题

### 3.1 资源与数据

- SSH：`matengyu@10.8.176.68:12022`；显式使用现有 `id_ed25519_neuroadapter_5090` 密钥后连接成功。
- GPU：2×RTX 5090，每卡 32,607 MiB；驱动 595.84。检查时分别占用约 13,022 / 6,805 MiB，存在运行任务。启动前重新检查资源，不终止现有任务。
- `/data` 剩余约 5.5 TB；系统内存 125 GiB，检查时约 91 GiB available。
- 只读探测的旧项目环境为 Python 3.11.16、PyTorch 2.11.0+cu128，编译架构包含 sm_120。这只说明存在可参考的运行时，不代表新项目已完成 GPU 运算、双卡通信或模型兼容验证。
- 已验证 NSD 图像 HDF5：`imgBrick=(73000,425,425,3)`，uint8。
- 已验证现有脑数据为 `lh_betas` / `rh_betas`，各 `(30000,163842)`，float32；它们是表面空间数据，不能直接用作本文体积空间输入。
- 目标体积 beta、new_test 索引、COCO caption 缓存与完整预训练权重仍需盘点和准备。此次限定目录内未找到目标 beta 不等于全服务器不存在。

### 3.2 源码与论文的差异

| 项目 | 已观察到的情况 | 处理与验收 |
| --- | --- | --- |
| 配置入口 | `configs/default.yaml` 不被训练入口读取 | 运行器输出完整命令和实际解析后的配置，不仅保存 YAML |
| 默认训练量 | 阶段一默认 batch=64、100 epochs；阶段二 bs=768、1,000,000 iterations | 显式覆盖到论文目标，并核验实际样本数与更新次数 |
| 阶段一调度 | 源码提供 OneCycleLR/LinearLR，论文写 warmup + cosine | 固定复现目标并记录调度差异；按论文实现调度时保留独立补丁与学习率轨迹 |
| 阶段一损失 | 代码包含文本 MSE，SoftCLIP 实现为软目标双向损失；论文公式与文字并未完整表达这些细节 | 建立逐项 loss 对照，以作者实现为可执行基线，差异明确披露；不能静默删项或改损失 |
| 跨卡对比学习 | 训练路径未见用于扩大损失负样本集合的跨卡特征 gather | 同时记录优化器有效 batch 和对比负样本 batch；若实现全局 512 对比集合，验证梯度与单批参考一致 |
| 阶段二权重 | 仅传入 `hierarchy_ar_ckpt` 才加载生成器预训练；存在非严格加载路径 | 保存来源、SHA-256、匹配/缺失/多余键及参数覆盖率；核心权重缺失不得按预训练微调开跑 |
| 梯度累积 | 阶段二循环直接累积 loss，未除以累积步数；bs 解析未包含累积倍数 | 修正平均梯度语义；以 `每卡 microbatch × 2 × accum = 80` 验证有效 batch 和更新计数 |
| 数据加载 | new_test shard 枚举需检查；数据集内部使用状态式迭代器 | 核对实际 shard、worker/rank 分片、样本覆盖与重复；避免伪造缺失 shard 或反复读取同一样本 |
| 推理解码入口 | notebook 含绝对路径、外部 `mindbridge_models` 依赖和部分未重复响应处理 | 转换为独立 CLI；固定顺序和预处理；逐项核验缺少第三次响应时的处理规则 |
| 四候选选优 | `recon4_infer.ipynb` 将真实图像传入 `original_image`；pipeline 用其 CLIP 特征选优 | 标为 GT-assisted best-of-4，与不使用 GT 选优的 N=1 分开报告 |
| 辅助特征 | recon4 notebook 有混合 MindEye2 blurry reconstruction 的单元 | 主实验关闭混合；MindHier† 另设配置、依赖、目录及结果行 |
| 指标预处理 | notebook 在指标前将图像缩到 256，独立 eval 入口未显式做同一步 | 冻结完整预处理链；作者 notebook 和独立脚本的差异作为预检项目，不能混用后宣称同协议 |

阶段一的小 batch 梯度累积不能自动得到 batch=512 的 SoftCLIP 负样本集合。优先通过 BF16、梯度检查点和双卡通信实现目标；如必须使用特征重计算/梯度缓存，需要先验证损失和梯度等价。若最终无法满足目标，明确标注资源适配版本，并单列其配置和结果。

## 4. 执行阶段与验收

### P0：独立仓库与可追溯基础设施

1. 在 `/data/matengyu/geyugong/mindhier-reproduction/` 建立 `main` 分支及项目 Git 仓库。
2. 固定作者提交 `dfcdfdcb60acad9f8e9a88785e5b2776f828f669`，记录英文论文哈希与代码来源。将运行源码作为无嵌套 `.git` 的版本快照纳入 `src/mindhier/`，保留许可证与来源说明；现有本地 `upstream/` 留作只读参考并排除在外层仓库之外。
3. 建立配置、脚本、实验日志、偏差表、指标表和产物清单。原始数据、环境、缓存、完整解码及大权重排除在普通 Git 对象之外。
4. 本地通过 SSH Git remote 获取服务器提交；网络不适合直接 fetch 时使用保留原提交 SHA 的 Git bundle。代码同步与大文件同步分别校验。
5. 由本地 `gh` 创建 private 仓库、核验可见性，使用本地 Git 推送。每次同步记录 server/local/GitHub 三个提交 SHA；不以重新复制后另作一次提交代替同步。

验收：三端同一提交；能查看来源及许可证；未误提交凭据、原始数据或超大产物；已有本地资料保持完整。

### P1：数据准备与协议冻结

1. 盘点所需 NSD 体积 beta、nsdgeneral mask、图像 ID、40-session 元数据和 captions。优先只读复用合格数据，缺少内容再补齐。
2. 核验 beta 生成、标准化和体素选择过程；Subject 1 预期输入 15,724 体素，其余受试者按各自 mask 确认。已有 surface beta 不做简单 reshape 或降维后充当论文数据。
3. 生成受试者级 manifest：NSD ID、COCO ID、session、trial、beta 行号、split、重复次数、文件来源与哈希。检查 train/test 图像 ID 交集为零。
4. 固定新版 1,000 张测试图及排序，记录每张实际响应次数。若存在不足三次响应，明确记录作者规则与实际可用响应平均之间的差异，禁止静默复制响应后声称取得三次独立测量。
5. 用训练集内的调试划分完成参数和代码验证。正式训练按冻结的完整训练协议重启；最终 checkpoint 预先固定为阶段一末轮和阶段二末次更新，测试集不用于挑 checkpoint 或调参。

验收：每个图像与 beta 精确对应；测试图像数量及排序确定；全部测试候选与 GT 均可经 ID 联接；数据量报告来自实际清单。

### P2：专用环境、权重和双卡可行性

1. 在新项目内部创建隔离环境和缓存；固定 torch、torchvision、CUDA 运行时、NumPy、transformers、webdataset 及指标模型版本。
2. 验证 RTX 5090 前向/反向、BF16、双进程 NCCL、检查点恢复；FSDP 与梯度检查点根据实测选择。
3. 验证 CLIP ViT-L/14、Switti、VQ-VAE 的来源和架构匹配。权重清单覆盖下载地址、revision、文件大小、SHA-256、加载报告。
4. 完成训练与推理中的必要补丁。使用合成小张量验证有效 batch 与梯度尺度，使用固定小批数据验证候选—样本排列和八项指标。

验收：两阶段各能完成真实模型的前向、反向、保存和恢复；每卡峰值显存可接受；核心预训练参数加载符合预期。

### P3：Subject 1 小规模端到端试运行

使用训练集内固定样本完成阶段一和阶段二各约 100–300 个更新，再在内部调试样本上解码 16–32 张图。该试运行用于验证，不报告为正式测试成绩。

必须检查：损失有限、目标冻结模块无梯度、层级与 mask 对应正确、保存后重载输出一致、断点恢复的 step/随机状态/调度正确、结果 ID 无缺失和错位、八项指标能够独立重算。比较同图与置乱配对作为指标 sanity check；核对分批特征提取与小规模参考实现的一致性。

记录每步耗时、每卡峰值显存和数据吞吐，据此估算完整训练时间。试运行结束后冻结主实验配置，再开始正式训练。

### P4：Subject 1 正式训练

- 阶段一完成 300 epochs，并保存逐项 MSE/SoftCLIP、学习率、梯度范数、实际 batch 和吞吐。
- 阶段二冻结阶段一末轮模型，从经过核验的 Switti 权重微调 9,000 次优化器更新，有效 batch=80；先试每卡 4 × 2 卡 × 累积 10 的候选配置，能否采用以显存和梯度验证为准。
- 训练主种子拟固定为 42，记录 Python/NumPy/CPU/CUDA/rank 随机状态。实际作者默认值和本次采用值均写入配置对照。
- checkpoint 保存模型、优化器、调度器、混合精度状态、随机状态、epoch/update、数据采样进度和配置。阶段一每 10 epochs、阶段二每 500 updates 保存里程碑，另保留可恢复的最新版本及最终版本；周期根据实测文件大小调整并记录。
- 每次运行绑定一个干净的代码提交。失败、OOM、NaN、重启和参数修改各自形成记录；改变实质配置时新建 run ID。

验收：正式预算全部完成，关键 checkpoint 可恢复，日志及配置完整；不把试运行 checkpoint 或测试集选出的最高分结果作为正式结果。

### P5：完整解码与八指标评估

对 Subject 1 的 1,000 张测试图生成并保存：

1. **N=1**：不使用真实测试图像参与生成或候选选择。用于报告纯 fMRI 推理能力。
2. **N=4、GT-assisted**：依据作者 notebook 的 CLIP 真实图像选优协议生成四个候选，保存全部候选、各自分数和选择索引；报告中明确其使用 GT 的环节。

作者 notebook 的参考推理参数为 seed=59、cfg=6、top_k=400、top_p=0.95、turn_on_cfg_start_si=1。实际 pipeline 内部是否按尺度覆盖这些数值、种子重置及批量形状效应均需验证，并把最终执行的值存档。N=1 单独运行；不未经验证就以 N=4 的第一张代替。

主实验不混入辅助低层重建。正式计时使用 CUDA 同步，区分模型预热、纯生成、四候选选择和保存耗时，记录推理硬件与 batch。论文的单张 RTX 4090 耗时不作为 RTX 5090 的硬性达标值。

验收：每个协议 1,000 个唯一测试 ID；N=4 共 4,000 个原始候选；保存的模型输出张量与 PNG 均可追溯；只加载保存结果即可重算八项指标。

### P6：扩展至四受试者与主表对照

按已冻结配置分别训练 Subject 2、5、7。先逐受试者计算指标，再对 Subject 1、2、5、7 等权平均；单受试者结果与四人平均分栏报告。双卡调度以 P3 显存结果为准：需要分片时两卡服务同一任务，不预设两张 32 GB 可以视为一张 64 GB。

先完成每受试者一个固定训练种子的主实验。资源允许后增加独立训练种子，报告种子均值与标准差；多次解码种子与多次训练种子分开统计。若报告 bootstrap 区间，说明其基于测试图像，不能冒充训练种子方差；论文 ± 的具体来源未核实前不作同口径误差比较。

### P7：核心消融与可选扩展

在 Subject 1 上依次复现 Table 2–4：单层特征、分层但仅末层监督、三种 CLIP 层映射、反向 fine-to-coarse 引导。每组固定数据、预算与评估协议，只改变目标因素。

MindHier† 作为独立扩展，需额外核验 MindEye2 低层预测的来源、训练划分和样本顺序，并独立保存融合前后结果。脑定位、VQA、THINGS-fMRI 和跨受试者实验属于进一步扩展，不作为此次 NSD 八指标主流程的先决条件。

## 5. 八项指标与论文参照值

以下定义以作者 `brainencoder/eval.py` 为起点，实施时冻结模型权重、特征层、resize/normalize/antialias、数值精度、样本顺序和聚合规则。AN2、AN5、Incep、CLIP 为二选一识别率，不是分类 top-k 准确率；Eff、SwAV 是相关距离，不是耗时或余弦距离。

| 指标 | 计算约定 | S1 论文参照 | 四人平均论文参照 |
| --- | --- | ---: | ---: |
| PixCorr ↑ | 配对图像像素 Pearson 相关系数均值 | 0.273 | 0.235 |
| SSIM ↑ | 灰度 SSIM，Gaussian σ=1.5、data_range=1、非样本协方差 | 0.394 | 0.381 |
| AN2 ↑ | AlexNet `features.4` 特征二选一识别 | 97.0% | 94.0% |
| AN5 ↑ | AlexNet `features.11` 特征二选一识别 | 99.3% | 98.4% |
| Incep ↑ | Inception-v3 `avgpool` 二选一识别 | 96.7% | 95.9% |
| CLIP ↑ | OpenAI CLIP ViT-L/14 图像 embedding 二选一识别 | 97.2% | 96.4% |
| Eff ↓ | EfficientNet-B1 `avgpool` 配对相关距离均值 | 0.598 | 0.606 |
| SwAV ↓ | SwAV ResNet-50 `avgpool` 配对相关距离均值 | 0.321 | 0.329 |

来源：英文原文 p.6 Table 1、不带 † 的 MindHier；p.16 Table S1。以上为论文参照值，当前没有本实验实测指标，也不预先承诺达到这些数值。

二选一指标采用全部其他图像构成的比较集合，记录相关矩阵方向、排除自身、严格大于比较和并列处理。采用分批提取特征避免将整个测试集一次放入 GPU，但必须保持完整 N×N 比较集合，不能把每个小批次的识别率平均当成正式结果。

## 6. 产物保留、日志和三端同步

建议结构：

```text
mindhier-reproduction/
  README.md
  PLAN_5090_ZH.md
  UPSTREAM.md
  src/mindhier/                 # 纳入主仓库的源码快照与修复
  configs/                     # 冻结配置、实际参数导出
  scripts/                     # 数据、训练、解码、评估、同步入口
  protocols/                   # 数据划分、指标及论文偏差表
  experiments/                 # 研究日志、失败与恢复记录
  reports/                     # 指标 CSV/JSON、曲线、固定示例、最终报告
  manifests/                   # 数据、权重、结果的路径/大小/SHA-256
  runs/<run_id>/               # checkpoint、完整日志、事件和资源采样
  artifacts/<run_id>/<subject>/
    n1/                        # 浮点输出与无损 PNG
    n4/candidates/             # 四个候选全部保留
    n4/selected/               # 选中结果
    selection.jsonl            # ID、分数、索引、seed、协议
    samples.csv                # NSD/COCO ID 与文件映射
  data/ pretrained/ .envs/ .cache/  # 大文件与本地运行环境
```

每个 run 至少记录：研究目的与假设、来源提交、数据清单哈希、完整命令、实际参数、依赖锁定文件、硬件、起止时间与时区、训练及推理种子、阶段/受试者、逐项损失、学习率、显存、耗时、异常、恢复链、输出哈希、结果与解释。日志用“实验目的—配置—操作—结果—异常—结论”的研究记录口吻；计划、观察和解释分开，缺失值标记待核验，不填造结果。

服务器保存完整 checkpoint、原始解码张量、所有 PNG、候选分数、评估特征和指标。至少把最终 checkpoint、全部测试解码及必要重算产物同步到本地并核验 SHA-256；GitHub 普通 Git 保存代码、记录、指标、清单和适量固定示例。大文件通过独立归档管理，不因为未进入 Git 而删除。若将大型归档纳入 GitHub，先核验 LFS/Release 的容量与访问条件，再确定传输方案。

提交按有意义的节点产生：

```text
docs(plan): define MindHier reproduction protocol
chore(repo): initialize reproducible experiment workspace
chore(env): pin Blackwell-compatible runtime
feat(data): validate NSD volume splits and manifests
fix(train): correct accumulation and checkpoint loading
feat(eval): add reproducible decoding and eight-metric evaluation
docs(experiments): record subject 1 training and evaluation
docs(results): report four-subject reproduction results
```

每次提交后先同步代码与轻量记录；阶段完成后同步大型结果。活跃日志在落盘后生成一致快照，checkpoint 原子写完后再传输，保存同步清单和校验结果。同步不得使用会清除本地独有 PDF、作者参考仓库或既有结果的删除选项。

## 7. 最终验收与工期估算

主复现完成的条件：四受试者正式训练预算完成；每人 1,000 张测试解码及四候选全部可追溯；两种选优协议分开；八指标可以离线重算；相对论文的逐项差值及协议偏差有说明；服务器、本地与 private GitHub 的代码提交一致；完整结果至少有服务器与本地两份经校验副本。

耗时在 P3 后计算：阶段一按实测每 epoch 时间×300，阶段二按实测每次优化器更新时间×9,000，解码按每图端到端时间×测试图数，并另计下载、特征预计算、评估、备份和资源等待。正式训练时间尚未测得，不将其他 GPU 或论文推理耗时外推为训练完成日期。

当前优先顺序：**P0 仓库 → P1 数据 → P2 环境与补丁 → P3 试运行 → P4/P5 Subject 1 → P6 四受试者 → P7 消融与扩展**。
