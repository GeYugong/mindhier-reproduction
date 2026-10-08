# 内部八指标与阶段二正式协议

## 验证范围

对已保存16个训练样本解码进行离线八指标计算，未调用测试图像。全部先bilinear/antialias缩放256，再遵循作者eval.py各指标resize/normalize。Eff采用当前专用环境DEFAULT实际对应的efficientnet_b1-c27df63c；SwAV按作者hubconf加载骨干并忽略projection_head/prototypes，分类fc不用于avgpool特征。权重哈希、逐指标特征、样本ID、分批对照结果全部保留。

GPU批量4与批量1的特征一致性两次未通过：默认设置AN5最大误差0.003059，关闭TF32后AN2最大误差0.040552。尚未确定根因，不放宽容差；原始证据在reports/pilot-metrics-attempt1/2。第三次CPU执行因SwAV预训练多出投影头而在严格审计处停止，证据attempt3。修正为仅允许明确的预训练头部多余键，并确认Eff使用作者DEFAULT对应文件，重新完整计算。

CPU FP32关闭MKLDNN的参考流程通过；所有模型的前4个GT分别batch1与batch4比较，最大绝对误差3.8147e-6。同图PixCorr/SSIM/二选一均为1，相关距离均为0；循环错位对照已记录，16图有限样本不要求恰为随机0.5。二选一完整16×16矩阵还与np.corrcoef参考实现比较一致。数值见reports/pilot-eight-metrics/result.json，只是pilot流程验证，不能与论文正式测试成绩直接比较。

## 正式协议

重新核对英文论文p.5–6 §3.3，h_k=M-floor(M(k-1)/K)，M=4,K=10；十尺度应依次使用脑层4,4,4,3,3,2,2,2,1,1，对应CLIP24,24,24,20,20,16,16,16,12,12。原训练正序属于与论文方向相反的行为，pilot仅作为工程验证。正式训练与推理均反转脑层列表，沿用已修复的尺度全局mask offset，从预训练重新初始化，不从pilot续训。

configs/stage2-paper-s01.json冻结batch80、9000更新、seed42、576 Lanczos后随机512裁剪、不翻转，完整27000训练trial确定性乱序。AdamW(.9,.95)、lr1e-4；weight decay=0、clip norm1、uncond=.1来自作者配置，空条件为全零beta经冻结脑编码器。VQ噪声None。warmup100、初始比例.005、余弦末比例.01为显式实施选择，论文未报告具体长度/比例，不声称完全复刻未公开参数。尚需实现正式数据/RNG恢复与训练入口，并核验更大microbatch吞吐；不能将此配置文件当作正式训练已启动。
