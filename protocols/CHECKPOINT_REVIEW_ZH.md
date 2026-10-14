# Checkpoint 自动评估与轮询决策

2026-10-14。用户授权在后续 checkpoint 后检查优化/收敛，并根据评估结果决定后续操作。本次为当前 Subject 1 Stage 2 增加独立、常驻、CPU 评估器；没有重启或改写正在执行的 06e95b8 训练。训练 PID 仍应以 child.pid 和真实命令为准。

## 已实施的测量

1. 每 20 秒检查 checkpoint.json 与 checkpoint.pt。只在完整 checkpoint 发布后评估，每个 checkpoint 只保存一份对应 logits 的评估；重复启动由进程核对及 flock 阻止。
2. 对齐 checkpoint.step，读取截至该步的训练日志，检查连续性与非有限值，并计算相邻 100/500 步窗口、完整 500 步块的均值和变化率、学习率与梯度范数。正在进行的更新不计入。
3. 复用原训练器每次保存的 reference_logits。固定输入是训练流 position=0，NSD ID 58173、beta row 5017、crop top=48/left=20，正常脑条件（unconditional=false）。这一个样本已参加训练，绝不是独立验证集。
4. 使用原 06e95b8 的数据和 VQ 代码、相同冻结 VQ 权重、相同 CUDA FP32/noise_std=None，对该图的目标 token 做一次性冻结。同一卡 GPU 0 两次编码得到完全一致的 2240 token；一次性准备峰值显存约 1.36 GiB。后续每次评估均为 CPU，不额外加载生成器、不重新前向、不占 GPU，也不修改训练 RNG。
5. 计算固定样本 teacher-forced token CE、top-1 准确率、预测熵与 10 个尺度的 CE/准确率。尺度值分列；整体 CE 按全部 token 等权，不能把 10 个尺度无权平均冒充整体指标。
6. mmap 读取 checkpoint，仅触及参考 logits 和必要元信息，核验格式、来源提交、配置哈希、step、next_sample、world size、seed 与 pilot 标志，并检查读取前后 inode/大小/mtime 和发布标记一致。这里没有每次重新计算整份约 26 GiB checkpoint 的 SHA，也没有冒充完整模型恢复验证。

## 决策规则

操作性阈值固定为：相邻两个完整 500 步窗口比较连续出现相对恶化至少 1%，或绝对相对变化小于 0.1%，触发审查。参考样本同样仅在连续三个相隔 500 步的 checkpoint 上做两次比较；缺失里程碑时不能假定确认。

- `continue`：最近 500 步训练 CE 改善超过 0.1%，无持续审查信号。
- `observe`：尚无确认信号，继续收集下一 checkpoint 的证据。
- `review_signal_disagreement`：训练或固定参考样本有持续停滞/恶化，需要对照分尺度、学习率、数据顺序和日志复核。
- `review_before_continuing`：两类信号都触发审查，轮询代理应在继续大量消耗训练预算前复核，并决定是否保留完整 checkpoint 后暂停。两类信号可能方向不同，必须读取具体 flags，不能只看 action。
- 评估失败、缺失日志、来源/配置不匹配、非有限值：状态为 `evaluation_failed`，不能把评估缺失当成模型健康。

这些阈值是明确记录的运行审查规则，不是论文报告或已经验证的早停标准。评估器不自行改学习率、选 best、替换编码器、终止训练或删除 checkpoint。轮询时依据这些证据作出继续、进一步诊断或在完整 checkpoint 后暂停的决策；单次波动和单个训练样本的停滞不直接触发停训。若决定改变固定预算，必须单独记录依据、配置偏差及可恢复状态。

当前实验仍没有独立验证集，不以训练 CE 下降、固定样本 token 准确率提高或 pilot 八指标推断未见图像重建质量；不使用正式 1000 测试图早停/选模。全量正式解码、八指标和 N1/N4 区分仍按既定流程执行。

## 已有 checkpoint 的补评估

对保留的 500、1000、1500、2000、2500、3000、3500 以及当前 3900 checkpoint 完成补评估。参考 CE 依次为 6.508226、6.441972、6.387798、6.388873、6.383105、6.324778、6.264870、6.253750。当前 3900 的决策为 continue。

2500 的固定单样本曾触发 plateau 提示，而其后 CE 重新下降，说明单样本提示需要复核，不能机械地据此早停。本次仅补评估已有正式 checkpoint，没有重跑 Stage 1、pilot、原门控测试或训练更新。

## 运行与复核

- 启动或核对唯一监测进程：`.envs/mindhier/bin/python scripts/launch_checkpoint_review.py`。
- 一次检查：`CUDA_VISIBLE_DEVICES= .envs/mindhier/bin/python scripts/review_stage2_checkpoints.py`；已有常驻进程时此命令会拒绝重复，应读取现有 JSON。
- 历史补评估：在没有常驻进程时加 `--backfill`；不重复训练。
- 结果：`runs/checkpoint-review/reviews/*.json`、`latest.json`、`index.json`。
- 健康状态：`monitor-status.json`、`monitor.pid`、`monitor.log`；同时核对真实命令和 cwd。
- 主机重启后由心跳检查先确认训练与监测进程，再用 launcher 恢复评估器；不能重复启动训练。监测状态过期超过 120 秒、evaluation_failed 或最新 checkpoint 长期没有评估都要调查。
- 新受试者/新 run 在启动前需绑定自己的训练来源、配置、参考输入及验证策略。本脚本严格绑定当前 S1 的来源/配置，不直接套用到其他受试者。

5 项回归测试覆盖截断 JSONL、checkpoint 边界、缺失/重复步骤、非有限值、持续恶化/单次波动、平台与缺失里程碑、CE/准确率/熵的数值定义。额外隔离进程测试模拟先后发布 100 和 200 checkpoint，并验证自动发现、防重复启动和同一 checkpoint 不重复产出；合成夹具仅位于独立 runs 目录，未触碰实际训练。

实现检查阶段产生的历史补评估明确标注 dirty 状态和当时脚本 SHA；正式常驻进程在提交后从干净主仓库启动，报告另存训练来源 SHA、监测提交及脚本 SHA。监测代码变更不改变当前训练的原始绑定。
