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
