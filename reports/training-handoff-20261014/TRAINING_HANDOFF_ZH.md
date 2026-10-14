# MindHier 当前训练交接记录

核验时间：2026-10-14 15:27:12 北京时间（UTC+8）。本记录是时间点快照，后续进展以服务器运行文件为准。原始核验数据见同目录 snapshot.json，最近100步记录见 metrics-last100.jsonl。

## 当前结论

Subject 1、seed42 的 Stage 2 正式训练正在运行，完成 3756/9000 次参数更新（41.73%）；进行中的 active_step 不计入已完成步数。Stage 1 已完成；正式测试解码和八指标尚未完成，不能称整个复现完成。范围为不带 MindEye2 低层辅助的 MindHier，S1 完成后再扩展 S2/5/7。

| 项目 | 本次实测 |
|---|---|
| Run ID | s01-stage2-paper-seed42 |
| 训练/监督 PID | 50309 / 50300，均在运行 |
| 已完成更新 | 3756 |
| 已完成样本曝光 | 300480（含跨epoch重复曝光，不是独立图片数） |
| 折合完整数据遍历 | 11.1289 epochs |
| 最新训练CE | 5.84165251 |
| 最近100步平均CE | 5.83452586 |
| 首步CE | 9.65914569 |
| 当前学习率 | 6.420245825e-05 |
| 当前梯度范数 | 0.46019152 |
| 最近10步耗时中位数 | 149.33 秒/更新 |
| 最近100步耗时均值 | 151.21 秒/更新 |
| 剩余更新 | 5244 |
| 剩余纯训练估算 | 9.18 天；不含后续解码、评估、备份、额外停机及检查点写入开销 |
| 最近检查点 | step 3700，27,876,378,698 bytes（约 25.96 GiB） |
| 非有限数值记录 | 0 条 |

CE下降只说明训练目标变化，不能替代测试重建效果。近期速度受共享CPU/GPU负载影响，估计不是承诺完成日期。

## 数据与模型

- NSD Subject1，1.8 mm native volume、nsdgeneral mask；beta矩阵30000×15724。
- 训练27000 trials / 9000独立图像；测试3000 trials / 1000独立图像，每图3次脑响应；训练/测试图像交集为0。
- 训练保留单trial；测试协议按图像汇总重复响应。测试集不用于调参、选seed、选checkpoint。
- Stage1：四层脑编码器，d_model=768，脑token序列长度100，图像token257、文本token77；对齐冻结CLIP ViT-L/14第12/16/20/24层。
- Stage2：Switti条件适配生成器，2,318,294,656个参数，30层、hidden=1920、30 heads、MLP ratio=4，条件维度768。脑编码器和VQ-VAE冻结。
- Switti匹配610个键，按参数量覆盖96.1189955%；其余已审计的条件适配参数按固定seed初始化，并非全部参数均直接来自预训练。审计见 manifests/switti-adaptation.json。
- VQ词表4096、z_channels=32、ch=160、share_quant_resi=4；输出512×512。
- 图像尺度patch网格：[1,2,3,4,6,9,13,18,24,32]，合计2240 tokens。
- 深到浅脑层映射：[4,4,4,3,3,2,2,2,1,1]，context_order=reverse。

## Stage1 已完成配置

- Run：s01-stage1-paper-seed42；300 epochs，每epoch drop_last后52更新，共15600更新。
- 有效batch=512，SoftCLIP负样本集合也为512；microbatch8，通过保持RNG的全批梯度缓存实现，物理GPU1。
- AdamW lr=1e-4，betas=(0.9,0.999)；一般参数weight_decay=0.01，排除衰减的参数组为0。
- MSE权重2e5、SoftCLIP温度0.005；保留作者文本MSE与完整patch/token SoftCLIP。
- warmup2 epochs，之后cosine到峰值学习率的0.01；BF16前向，FP32目标缓存和损失。
- 每10epochs保留检查点，固定第300epoch为最终脑编码器，不通过测试结果选择。
- 最终完整文件3,687,716,107 bytes，双端校验SHA256：f03f44ec732c2304f05efd03950f133239de266d9a9fec8866a0877e105f5b13。

## Stage2 正式配置

- seed42；9000次更新，有效batch80，microbatch4×累积20。每个microbatch梯度显式按4/80在CPU累加，完整batch之后更新一次。
- 总预算720000次trial曝光，约26.6667个epoch；每epoch27000/80=337.5次更新当量，batch可跨epoch边界，不是每轮固定截断为338步。
- 27000条训练trial每epoch完整确定性乱序；曝光位置决定裁剪与无条件dropout，并记录样本ID、beta行、crop与epoch。
- 图像先短边576 PIL Lanczos，再512×512随机裁剪；无水平翻转。
- AdamW lr=1e-4，betas=(0.9,0.95)，weight_decay=0，foreach=False；全局梯度clip norm=1。
- 前100更新线性warmup，初始比例0.005（5e-7）；随后cosine，终值比例0.01（1e-6）。
- 无条件概率0.1，零beta经过冻结脑编码器；VQ noise_std=None。
- batch及2240 tokens上的平均交叉熵；Stage2从Switti预训练重新初始化开始，未从pilot/validation续作正式结果。
- 单GPU0，world_size=1；FSDP实际NO_SHARD，使用CPU offload；每个block的非重入checkpoint_wrapper在FSDP内部；BF16 blocks、FP32 root、CPU FP32 master参数和Adam状态。
- torch CPU threads=4，autocast cache_enabled=False，单进程显存上限比例0.85。
- 当前没有启用Apex RMSNorm；日志显示使用普通实现，fused MLP/add-LN启用数为0/30。

## 资源和环境

本次GPU快照：

```text
index, name, driver_version, memory.total [MiB], memory.used [MiB], utilization.gpu [%], temperature.gpu
0, NVIDIA GeForce RTX 5090, 595.99.02, 32607 MiB, 25149 MiB, 59 %, 65
1, NVIDIA GeForce RTX 5090, 595.99.02, 32607 MiB, 29125 MiB, 100 %, 64
```

本次GPU进程显存：

```text
gpu_uuid, pid, used_gpu_memory [MiB]
GPU-3bbc128e-9d16-a851-f900-ef1ba565565a, 50309, 21914 MiB
GPU-3bbc128e-9d16-a851-f900-ef1ba565565a, 82321, 3140 MiB
GPU-8576fab9-b2f2-0693-5a93-5437992708fb, 28783, 8112 MiB
GPU-8576fab9-b2f2-0693-5a93-5437992708fb, 62793, 9826 MiB
GPU-8576fab9-b2f2-0693-5a93-5437992708fb, 700357, 3668 MiB
GPU-8576fab9-b2f2-0693-5a93-5437992708fb, 1066851, 7460 MiB
```

GPU0中的50309是本训练，其余列出的GPU进程属于其他任务。GPU1仍为100%利用率，不能视为空闲。单卡安排由共享资源约束决定；双卡基础通信测试通过，不等于完整双卡训练及恢复验证通过。

- Python：3.11.16；包版本：{"torch": "2.11.0+cu128", "torchvision": "0.26.0+cu128", "numpy": "1.26.4", "transformers": "4.44.2", "safetensors": "0.6.2"}。
- 项目环境 .envs/mindhier 为venv，读取已有基础环境的system site packages并使用项目依赖覆盖；不是完全独立复制全部库，基础环境未由本项目修改。
- 内存/Swap实测原始输出见snapshot；内存总量约125.1GiB，可用量随共享负载变化。Swap接近用满，不据此单独判定训练失败。
- 服务器项目所在盘可用约4937.6GiB。
- 本地D盘本轮查询可用28,570,931,200 bytes（26.61GiB），已不同于旧自动化提示中的7GB；仅略大于一份检查点，保留文件和.partial，不进行无余量的大型复制。正式检查点尚未完成本地副本校验。

## 启动、恢复及代码身份

- 正式首次启动：2026-10-08 02:55:12 UTC，即北京时间10月8日10:55:12。
- 训练源码固定：06e95b83a14b07fef51ebdb4c299b6df9914bb34。
- 本记录生成前主仓库HEAD：cc5536d81303c2d3078db556c699da2170cfbe27；后续本次文档提交不改变运行源码。
- 作者源码来源固定：dfcdfdcb60acad9f8e9a88785e5b2776f828f669。
- 10月12日15:59:10 UTC主机重启，原因未知；当时完成3047步、3048步未完成。先归档原始证据，再从step3000恢复，重放47步。归档保留在 runs/pipeline/formal-interruption-20261012T164933Z。
- 恢复使用原提交独立worktree，模型、Adam、scheduler、RNG和下一样本检查通过；reference logits最大误差0。恢复证据 reports/stage2-reboot-resume-3000，日志 experiments/2026-10-12-stage2-reboot-resume.md。
- 本次仍在运行，无formal退出码；validate和restore门控退出码均为0。stdout仅启动警告、监督日志为空不能用于判断停滞，应联合status/metrics时间与进程判断。

当前进程的命令记录（不是现在需要再次执行的指令）：

```bash
cd /data/matengyu/geyugong/mindhier-reproduction/runs/recovery-source-06e95b8
.envs/mindhier/bin/python -u scripts/train_stage2_formal.py --run s01-stage2-paper-seed42 --steps 9000 --microbatch 4 --resume
```

监督入口：/data/matengyu/geyugong/mindhier-reproduction/runs/stage2-formal-supervisor/resume-3000.py。该监督脚本为本次从3000恢复专用，不能在新中断时不审查就重新使用。

## 检查点和证据位置

服务器主目录：/data/matengyu/geyugong/mindhier-reproduction

| 相对主目录路径 | 内容 |
|---|---|
| runs/s01-stage2-paper-seed42/status.json | 已完成步、活动步、PID、恢复验证 |
| runs/s01-stage2-paper-seed42/metrics.jsonl | 每次更新的CE、LR、梯度范数、秒数 |
| runs/s01-stage2-paper-seed42/samples.jsonl | 每步80条样本曝光、随机裁剪、无条件标志 |
| runs/s01-stage2-paper-seed42/config.json | 实际运行配置，microbatch已覆盖为4 |
| runs/s01-stage2-paper-seed42/checkpoint.pt | 最新完整可恢复检查点 |
| runs/s01-stage2-paper-seed42/checkpoint.json | 步数、大小、配置哈希 |
| runs/s01-stage2-paper-seed42/checkpoint-step-*.pt | 500步间隔快照，当前500至3500 |
| runs/s01-stage2-paper-seed42/stdout-formal.log | 启动与异常输出 |
| runs/s01-stage2-paper-seed42/child.pid | 训练PID |
| runs/stage2-formal-supervisor/supervisor.pid | 监督PID |
| runs/stage2-formal-supervisor/exit-codes.jsonl | 门控及正式进程退出记录 |
| runs/stage2-formal-supervisor.log | 监督输出 |
| manifests/environment-freeze.txt | 依赖版本清单 |
| manifests/assets-verified.json | 权重及资产校验清单 |
| reports/stage2-formal-start | 正式启动与门控证据 |
| reports/stage2-data-replay.json | 确定性数据回放验证 |
| reports/stage2-microbatch4-average.json | 平均梯度验证 |

检查点首步和每100步原子更新latest，每500步保留快照。格式fsdp_world1_cpu_flat_v1，包含模型参数/buffers、Adam、scheduler、CPU/CUDA/Python/NumPy RNG、step/next_sample/spec、配置哈希、源码提交及参考logits。禁止默认full state_dict在GPU上汇总。

当前配置哈希：e08b20d94e6981ee6c3abdeb8b2d2cd9bd13e82f62ca8581ae3df82ca43b5a57。本轮核对最新检查点文件大小及元数据，没有对27.9GB文件重新计算SHA或加载恢复；不要把大小核对写成完整SHA验证。

已验证的step3000完整检查点SHA256：3a1f8c07efd3413521aeb9a88d9745cb75eab02fa2bf094ac46bf54473565ff1。该哈希不适用于当前step3700。

未来恢复必须先确认训练已停止、归档证据，再使用原06e95b8及相同FSDP包装核验最新checkpoint；严格配置/版本检查不可跳过。切换双卡需单独转换和验证，不能直接更改world_size续跑。

## 只读检查与连接

```powershell
ssh -o BatchMode=yes -o IdentitiesOnly=yes -i C:/Users/gyg17/.ssh/id_ed25519_neuroadapter_5090 -p 12022 matengyu@10.8.176.68
```

```bash
cd /data/matengyu/geyugong/mindhier-reproduction
.envs/mindhier/bin/python scripts/status.py
nvidia-smi
```

本地目录：D:/0code/Research/x_neuroadapter/mindhier-reproduction。GitHub私有仓库：GeYugong/mindhier-reproduction。同步流程为服务器提交→本地fetch server/ff-only→本地gh凭据push→三端SHA核验。私钥内容不进入文档或仓库。

## 已完成、偏差与剩余工作

已完成Stage1正式训练、Stage2的100步工程pilot、16张训练样本解码、八指标CPU FP32 sanity，以及正式启动门控与本次恢复。pilot只用于工程验证，其层级方向/数据等配置与正式训练不同，不能列为正式论文结果，也不需要重跑。

Stage1最终检查点、pilot完整检查点及16图解码产物（50文件）已有双端校验记录。pilot独立FP32生成器导出在服务器完整，本地generator-fp32.pt.partial保留。正式训练大检查点尚无已验证本地副本。

显式实施差异：warmup及cosine末值为记录过的选择；Stage1按实际27000条trial/drop_last定义epoch；caption按COCO annotation顺序取前三条，未取得作者curated caption文件；BF16/FP32、梯度缓存、CPU offload属于硬件适配。configs/stage2-paper-s01.json中的microbatch/status及早期PLAN/IMPLEMENTATION_NOTES含旧占位或计划描述，应结合实际run/config、status及后续实验记录解读，不能修改绑定版本配置来消除这些文字。

剩余：完成S1正式9000步→固定最终checkpoint→1000测试ID的独立N1解码与N4 GT-assisted解码，保留全部4000候选及选择分数/索引→PixCorr、SSIM、AN2、AN5、Incep、CLIP、Eff、SwAV八指标→最终产物双端SHA校验→扩展S2/5/7并报告四人均值。N1不使用GT选优，N4明确使用GT CLIP选优，不混作同一结果。

正式解码参数须以最终冻结执行配置为准；作者参考seed59、cfg6、top_k400、top_p0.95、CFG起始尺度1。八指标CPU FP32参考已验证，但正式测试八指标尚未产生。只有完整训练、解码、评估及结果保留完成后，才能宣布相应复现范围完成。
