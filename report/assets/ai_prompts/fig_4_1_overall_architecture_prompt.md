请生成一张学术论文风格的网络结构图，白底、矢量感、中文标注清晰。
主题：Edge-Guided BTS 单目深度估计总体结构图。
必须包含：
1. 输入 RGB 图像；
2. DenseNet121 编码器；
3. BTS 解码器和三尺度局部平面指导分支：lpg8x8、lpg4x4、lpg2x2；
4. 浅层 skip0 特征分出一个轻量边缘分支；
5. 边缘分支经过卷积得到 edge logits，再经过 sigmoid 得到 edge probability；
6. edge probability 与末端 upconv1 特征发生乘性调制；
7. 调制后的特征与 reduc1x1、多尺度深度结果、edge probability 共同拼接；
8. 最终输出 depth map。
风格要求：结构清晰、箭头明确、模块颜色克制、适合课程报告插图。
