# 正式CPU评估SwAV加载修复

N1前七项指标与特征成功保存；SwAV权重反序列化时因CUDA存储标签且CUDA_VISIBLE_DEVICES为空而失败。不是训练/解码失败。原失败状态、stdout和退出码已保留；五份特征NPZ SHA已冻结。

在所有torch.load权重/结果读取上显式map_location=cpu，不改变权重内容、预处理或指标。新增显式--resume，仅复用经过原始SHA核验、已有指标和NPZ同时存在的完整特征阶段；不重跑训练、解码或选优。N1恢复后只新增SwAV特征和指标；像素指标再次计算以保留同一预处理链。N4尚未启动，使用修正版从头评估。报告保留初始提交与resume提交。
