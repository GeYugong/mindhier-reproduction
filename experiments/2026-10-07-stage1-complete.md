# Subject 1 阶段一完成与阶段二加载预检

正式训练完成 300 epochs / 15600 次优化器更新；最后一步 loss=3.740126896649599，峰值训练分配显存 6.1235 GiB。全部逐步指标有限且步序连续，stdout 未发现 NaN、OOM 或异常堆栈。最终检查点已在 CPU 反序列化并核验模型张量有限，恢复位置为 epoch=300、batch=0。最终 SHA-256 为 f03f44ec732c2304f05efd03950f133239de266d9a9fec8866a0877e105f5b13，大小 3687716107 bytes。

完整原始配置、训练日志、指标及完成状态归档在 reports/stage1-final。最终检查点的本地传输正在进行，尚未确认双副本校验。

阶段二新增实际加载预检：严格限定可重新初始化的 62 个条件适配张量，其余 Switti 权重必须匹配既有审计；新条件层保留作者 init_weights(init_std=-1) 的 seed=42 初始化。VQ-VAE 与阶段一编码器均要求 strict=True。全分辨率合成输入前向只验证计算路径，不构成真实训练、反向、恢复或端到端验收。后续仍须显存适配、真实训练集试运行、9000 次正式更新、解码及八指标。

第一次阶段二加载预检被有限值断言中止：attention mask 的负无穷是屏蔽语义所需，不代表权重损坏。原始异常保存在 reports/stage2-preflight-attempt1；修复为 attention mask 与模型构造值精确相等，其余张量仍检查有限值。

最终检查点已完整传输到本地 artifacts/s01-stage1-paper-seed42/epoch-300.pt；本地 Get-FileHash SHA256 与服务器相同，双副本核验完成。

第二次预检完成 Switti 610 项加载、96.1189955% 参数覆盖，以及 VQ-VAE/脑编码器 strict 加载。全分辨率合成前向触及 45% 显存限额，OOM 原始证据见 reports/stage2-preflight-attempt2。分配13.43 GiB、保留未分配690.29 MiB；关闭 autocast 权重缓存重试，保持 FP32 参数与 BF16 运算，不调整模型结构或训练协议。
