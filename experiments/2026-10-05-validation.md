---
date: 2026-10-05
timezone: Asia/Shanghai
status: running
---
# 数据与训练实现验证

Subject 1 train/test 图像 ID 无交集。体积 beta 的全部数值已检查为有限值。CSV 使用 LF 行尾，以便服务器、本地和 Git 对象的哈希一致。

梯度缓存与保留全部计算图的参考实现对比通过：关闭 dropout 时最大梯度绝对误差 2.5332e-7；启用 dropout 并回放随机状态时为 1.1921e-7。对比损失在完整批次上计算，跨 microbatch 的负样本未丢失。

两张 RTX 5090 的小规模 BF16 反向与 NCCL all_reduce 均通过。rank 0/1 的输入分别为 1/2，结果均为 3。输出见 reports/gpu-probe.jsonl。

冻结 CLIP 的 FP32 特征缓存正在进行。训练流水线将先验证 batch=512 的显存与断点恢复，再完成训练集内的 100 次试运行，最后启动固定种子的正式阶段一训练。各阶段结果均由实际日志填写。

## 首次真实更新与恢复门禁

307292160 参数的阶段一模型已完成一次 batch=512 更新，loss=29.70017147，峰值显存 5.8125 GiB；首步在共享 GPU 与并行特征缓存条件下耗时 67.56 秒，不能直接视为正式训练稳态速度。已保存约 3.69 GB 的完整 checkpoint。

自动恢复首次被干净工作树检查拦截：同时新增的 Switti 审计文件尚未提交。该次没有继续更新模型，保留 checkpoint；审计提交后从 step=1 恢复。后续启动配置按 step 分别留档，避免覆盖恢复前配置。

Switti 预训练审计：30 层、宽度 1920，目标参数 2318294656；可直接匹配目标参数的 96.119%。不匹配集中在原 2048 维文本上下文到 768 维 fMRI 上下文的投影与归一化，另有新的 fMRI pooler。详细键列表见 manifests/switti-adaptation.json；阶段二训练前仍须验证真实加载和显存。

每小时检查已由本地 Codex 创建，任务 ID 为 mindhier，状态 ACTIVE。检查进程、日志、损失、checkpoint、资源及三端同步；仅在关键变化、完成、失败或需要处理时通知。

## 完整恢复检查通过

3 次 batch=512 更新已完成，实际损失依次为 29.70017147, 29.67138577, 29.44947278。step=1 保存后恢复到 step=3，峰值显存 6.1185 GiB。原始输出、逐步指标、启动配置及 checkpoint 元数据已归档到 reports/stage1-memory-resume/；完整权重保留在 runs/s01-stage1-memory-resume-check/latest.pt。该结果仅用于数值、显存与恢复验证。
