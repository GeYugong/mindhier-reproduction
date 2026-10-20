# 8500步S1正式选优与八指标

N1的1000张和N4的4000候选均完整落盘、正常退出，全部浮点张量SHA、形状、有限值与[0,1]范围检查通过；1000唯一ID与排序对齐。保留原始候选，不重跑推理或训练。

GT-assisted选优依据作者recon4_infer.ipynb与pipeline_recs_persample.py的全局CLIP ViT-L/14余弦、首个argmax并列规则。notebook引用的mindbridge_models未随仓库附带；从MindBridge公开源核验Clipper(hidden_state=False,norm_embs=True)预处理并保留只读来源快照及SHA：bicubic224、antialias=None、center crop、CLIP normalize。来源https://raw.githubusercontent.com/littlepure2333/MindBridge/main/src/models.py 。未执行下载源代码。作者all_images.pt不可得，GT由原NSD图按notebook内准备代码PIL bilinear resize512+ToTensor重建，明确记录此来源差异。选优采用GPU FP32、TF32关闭；不声称与作者未固定精度实现逐位一致。保留四分数、索引和被选浮点图；原四候选不删。

正式指标沿用已通过pilot数值sanity的CPU FP32（MKLDNN关闭）与FP64相关系数；先全部GT/重建resize256 antialias，再按原约定指标尺寸与归一化。八指标包括PixCorr、SSIM、AN2、AN5、Incep、CLIP、Eff、SwAV。N1和GT-assisted N4分别输出，特征全部存盘可重算。AN2/AN5/Incep/CLIP全1000x1000相关矩阵、每图999干扰项，严格比较并排除自身，分母999000，不能按特征batch平均识别率。16样本np.corrcoef独立子集对照只作算式校验，不替代全1000指标。

基础算式测试通过：11样本独立标量全比较一致、identity、精确可表示并列计0。严格浮点比较不引入人为epsilon，任意数学相等向量在BLAS舍入下可能有机器精度差异。每类特征保留首4图batch1/4数值一致性、identity与shift检查；这不是调参依据。正式输出阶段才填写指标，当前不把pilot数值当结果。

训练8500/9000是用户指定预算偏差。固定8500最终权重，不以测试指标挑检查点，不扩展其他受试者。
