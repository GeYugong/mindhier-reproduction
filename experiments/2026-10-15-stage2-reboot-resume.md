# Stage 2 再次主机重启后的 4100 步恢复

## 观察与归档

2026-10-15 01:22 UTC 检查发现训练、监督及 checkpoint 评估器均已退出。主机记录显示当地时间 08:28 关机、09:01 启动（UTC+8），本次 boot ID 为 a274c7b7-d381-44fb-8b6e-10f854e1c547。关机发起者及原因未知；内核日志无读取权限。原训练最后完成 4180 更新，第 4181 更新仅处理部分微样本；没有正式退出码，stdout 未发现 NaN、CUDA OOM 或异常堆栈。主机重启导致进程消失，不将其解释为模型不收敛。

所有恢复写操作前，原 status、metrics、samples、checkpoint 元信息、stdout、监督和评估日志及进程/主机证据已复制并计算 SHA，保留在 runs/pipeline/formal-interruption-20261015T012251Z。原始 4101–4180 记录完整保留。完整 4100 checkpoint 以硬链接保留在该目录；这是同一服务器文件系统中的原件保留，不是异机备份。

轻量原始证据压缩包为 runs/pipeline/formal-interruption-20261015T012251Z-raw-evidence.tar.gz，4,920,400 bytes，服务器及本地 SHA-256 均为 ce110b97fba3c0fd1c44029330de9ae8930e353ca4a010aa0acbb89edabf5172。压缩包不含 checkpoint，也不含后来生成的恢复结果。本地 D 盘约 26.5 GiB 空闲，本次未复制大型 checkpoint，也未删除任何结果。

## 文件与实际恢复核验

4100 checkpoint 为 27,876,378,698 bytes，整份 SHA-256 为 2cea0f8f526633063f4d2e0496b395e294fd782b642e9f1f4f0805cb52f2d249。格式仍为 fsdp_world1_cpu_flat_v1，配置哈希为 e08b20d94e6981ee6c3abdeb8b2d2cd9bd13e82f62ca8581ae3df82ca43b5a57。CPU 核验参数及优化器张量均有限，调度器 step=4100、next_sample=328000，保存的参考 logits SHA 与既有 4100 评估一致。文件核验报告本身不宣称执行完整模型恢复。

随后使用干净 runs/recovery-source-06e95b8，严格绑定原提交 06e95b83a14b07fef51ebdb4c299b6df9914bb34 和原 FSDP 包装，执行原训练入口 --resume。原代码实际恢复检查已通过，resume_verified=true、restored_step=4100、参考 logits 最大绝对误差=0.0。原训练代码的配置、来源、模型参数/buffers、Adam、调度器、RNG 和 next_sample_spec 核验均保留，未跳过门控，也没有重新运行已完成的初始门控试验。

专用监督脚本 runs/stage2-formal-supervisor/resume-4100.py 使用互斥锁及真实进程检查，并核验归档、checkpoint 和活跃日志后，将活跃 metrics/samples 截至 4100。原版 resume-3000.py 未复用或修改。监督 PID=20284，训练 PID=20304；真实 PID、命令和 cwd 仍须逐次核实。退出后才会添加 formal_resume_4100 记录，当前没有该退出记录。科学配置及 9000 更新预算不变。

## 恢复后的检查与决策

2026-10-15T01:32:48.430254+00:00，已重新完成至 4101 更新。恢复后已完成的 1 步与原记录逐条比较：每步全部 80 个样本的身份、裁剪和无条件标记一致，学习率一致。loss 和梯度差异实测值见 reports/stage2-reboot-resume-4100/replay-check.json；不能据此把尚未完成的整段 80 步称作已验证。活跃 metrics 与 samples 均从 1 连续至当前完成步数，无重复或缺口。

CPU checkpoint 评估器已恢复，PID=22537，状态 watching，最近状态更新在 120 秒内。它沿用冻结目标和既有 4100 评估，不重新准备目标或占用 GPU。4100 训练最近 500 步均值从 5.874344 降至 5.826269，约改善 0.8184%；固定训练参考 CE=6.254041，较 4000 单次上升约 0.5012%。连续 500 步里程碑没有触发持续恶化/停滞，决策为 continue；全局 check_convergence.py 也无 review_flags。

此处固定参考仅一个已见训练样本，teacher-forced CE、准确率和训练 loss 不证明独立泛化或自由生成质量。没有使用正式 1000 测试图调参、早停或选模。正式训练、全量 N1/N4 解码及八指标仍未完成，不称复现完成。

## 后续

继续核对 4101–4180 重放及随后 4200 完整 checkpoint 与自动评估；本次没有为缩短等待而强制额外保存或改变保存间隔。后续故障不得盲用任何固定步数恢复脚本，应先归档、核验当前 checkpoint，再绑定原版本恢复。监测/文档主仓库提交变化不改变运行源码。
