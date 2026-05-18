请生成一张学术论文风格机制图，白底、中文标注。
主题：边缘分支与深度特征融合机制示意图。
必须展示：
1. skip0 特征进入边缘分支；
2. 得到 edge logits；
3. edge logits 经 sigmoid 变为 edge probability；
4. edge probability 对 upconv1 特征进行乘性调制 F' = F * (1 + P_edge)；
5. 调制后特征与 reduc1x1、lpg8x8、lpg4x4、lpg2x2、edge probability 一起拼接；
6. 进入最终卷积层输出深度图。
要求突出“边界位置被强化”的含义。
