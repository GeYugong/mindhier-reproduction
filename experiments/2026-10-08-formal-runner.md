# 正式训练入口与门控验证

实现完整27000 trial的按epoch确定性乱序，按样本曝光序号确定随机裁剪与10%无条件选择；跨epoch批次不drop，恢复只需已完成更新×80定位并校验下一样本spec。保留每次曝光的ID、beta行、裁剪坐标和无条件标记。冻结FMRI/VQ编码器，逐microbatch生成真实随机裁剪VQ目标，脑层反转遵循已冻结深到浅映射。首轮预备仅使用固定样本检查恢复输出。

microbatch=4、20次累积实现有效batch80；累加以样本数4/80加权，更新前CPU全参数norm1裁剪。lr_ratio按已完成更新定义，首步lr5e-7，第101步达1e-4，第9000步达1e-6。Adam状态、scheduler、CPU/CUDA/Python/NumPy RNG、采样位置、配置哈希、源提交和固定参考logits随CPU flat checkpoint保存。

门控顺序：独立验证目录从Switti新初始化完成一次真实batch80更新并保存；新进程恢复全部状态并继续第二次更新；只有两次正常退出、resume_verified且metrics严格为1,2时，才从Switti再次新初始化正式9000步。该验证涵盖新数据流、裁剪、无条件训练、反向层级、调度、clip及较大microbatch，不重复此前固定80样本pilot。正式检查点首步/每100步写latest，每500步保留快照，不删除旧结果。

当前仅实现门控，实际运行结果以后续status/metrics/exit-code证据为准，不能在验证结束前声称正式训练启动。较大microbatch仅改变执行组织，不改变平均batch80与科学协议。默认GPU0当前可用，不使用其他项目占用GPU1。
