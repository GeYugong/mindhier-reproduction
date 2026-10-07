# 阶段二真实样本反向与 CPU offload 预检

阶段一已完成，本轮仅验证阶段二单个训练样本的前向、反向和 AdamW 更新，不属于 batch=80 正式训练。使用训练 manifest 首行的真实图像和 beta、最终脑编码器以及严格加载的 VQ-VAE 生成 teacher forcing 输入；图像固定 bicubic 512、noise_std=0，仅为显存探测。正式增强协议尚未冻结。

共享 GPU 0/1 检查时分别占用约15/25 GiB，采用 GPU 0、逐 transformer block FSDP CPU offload、activation checkpointing 和无 autocast 权重缓存。world_size=1，不宣称已验证双卡分片或全batch平均梯度。AdamW FP32参数，BF16前向，beta=(0.9,0.95)，lr=1e-4，weight_decay=0.01仅为本次预检配置。

待验证：有效batch=80平均梯度、完整checkpoint保存恢复、训练集内试运行与端到端指标。当前不得启动9000次正式训练或报告测试指标。日志位于 runs/stage2-backward-probe。

第一次真实样本反向在 13.48 GiB 进程显存限额处 OOM（已分配13.22 GiB），完整异常见 reports/stage2-backward-attempt1。第二次将 transformer block 的 FSDP 参数计算副本设为 BF16、梯度归约为 FP32，root 保持 FP32 以兼容作者显式关闭 autocast 的输入/输出投影。CPU master参数和 AdamW 状态仍为 FP32。新运行目录 runs/stage2-backward-probe-mp，旧结果保留。
