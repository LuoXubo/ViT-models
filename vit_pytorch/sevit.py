import torch
from torch import nn 



class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        """ SE注意力机制,输入x。输入输出特征图不变
            1.squeeze: 全局池化 (batch,channel,height,width) -> (batch,channel,1,1) ==> (batch,channel)
            2.excitaton: 全连接or卷积核为1的卷积(batch,channel)->(batch,channel//reduction)-> (batch,channel) ==> (batch,channel,1,1) 输出y
            3.scale: 完成对通道维度上原始特征的标定 y = x*y 输出维度和输入维度相同

        :param channel: 输入特征图的通道数
        :param reduction: 特征图通道的降低倍数
        """
        super(SELayer, self).__init__()
        # 自适应全局平均池化,即，每个通道进行平均池化，使输出特征图长宽为1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 全连接的excitation
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
        # 卷积网络的excitation
        # 特征图变化：
        # (2,512,1,1) -> (2,512,1,1) -> (2,512,1,1)
        self.fc2 = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # (batch,channel,height,width) (2,512,8,8)
        b, c, _, _ = x.size()
        # 全局平均池化 (2,512,8,8) -> (2,512,1,1) -> (2,512)
        y = self.avg_pool(x).view(b, c)
        # (2,512) -> (2,512//reducation) -> (2,512) -> (2,512,1,1)
        y = self.fc(y).view(b, c, 1, 1)
        # (2,512,8,8)* (2,512,1,1) -> (2,512,8,8)
        pro = x * y
        return x * y


