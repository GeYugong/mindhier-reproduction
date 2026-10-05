# 实施配置与偏差

- Subject 1 官方体积数据为 30000×15724，训练 27000 trials / 9000 unique images，测试 3000 trials / 1000 unique images，每图 3 次响应。以实际清单为准，论文训练数量 30000 不直接作为去除测试后的样本数。
- 缓存冻结 CLIP 的四层投影 patch 特征及前三条 COCO caption 的完整 token 特征，维持作者训练输入和损失的结构。CLIP 缓存采用 FP32，训练前向 BF16、损失 FP32；这是固定的数值精度设置。
- 原作者训练代码额外包含文本 MSE 与完整 patch/token 维度的 SoftCLIP。本实验保留这些代码行为，不将论文公式简写当作删除损失项的依据。
- 阶段一在物理 GPU 1 上使用 RNG 回放的全批梯度缓存，优化器 batch 和 SoftCLIP 负样本集合均为 512。两张 GPU 已完成 BF16 反向与 NCCL all_reduce 检查；单卡安排由当前共享资源占用决定。
- 先进行 3 次 batch=512 显存/恢复检查，再进行训练集内 100 次小规模更新，最后从固定种子重新初始化并训练完整 300 epochs。试运行和正式运行目录分离。
- 阶段一每轮遍历真实训练清单的确定性乱序，drop_last 后每轮 52 次更新，共 15600 次更新。与原代码按 30000 的约数定义 resampling epoch 有区别，实际曝光数在配置及日志中记录。
- warmup=2 epochs；余弦最低学习率比例 0.01。warmup 长度参考作者 OneCycle 的 2-epoch 设置，最低比例属于本次显式配置，论文未明确给出。
- COCO 原始 caption 按 annotation 文件顺序取前三条，并按作者的 minibatch 序号循环。未取得作者 curated caption 文件，因此将该项列为数据文本监督差异。
- 正式 checkpoint 固定为最后一轮；不使用官方测试集选择模型、调学习率或挑训练种子。
- 阶段二仍需完成 Switti 的条件维度适配审计、权重覆盖率和显存验证；阶段一结束不等于整个复现结束。
