import torch.nn as nn
import numpy as np

class InvertedResidual(nn.Module):
    def __init__(self, inp, outp, factor,stride,batch = False):
        super(InvertedResidual, self).__init__()

        hiddim = round(inp*factor)

        if batch:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hiddim, 1,1,0, dias=False ),
                nn.BatchNorm2d(hiddim),
                nn.ReLU(inplace = True),

                nn.Conv2d(hiddim, hiddim,3,stride,0,groups = hiddim, bias=False),
                nn.BatchNorm2d(hiddim),
                nn.ReLU(inplace = True),

                nn.Conv2d(hiddim, inp, 1,1,0, bias=False),
                nn.BatchNorm2d(outp),
            )
        else :
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hiddim, 1, 1, 0, bias=False),
                nn.ReLU(inplace=True),

                nn.Conv2d(hiddim, hiddim, 3, stride, 0, groups=hiddim, bias=False),
                nn.ReLU(inplace=True),

                nn.Conv2d(hiddim, outp, 1, 1, 0, bias=False),
            )

    def forward(self, x):
        return self.conv(x) + x


class MobileNet(nn.Module):
    def __int__(self,n_class = 1000, input_size = 224, batch = False):
        super(MobileNet,self).__init__()
        input_channel = 32
        output_channel = 1280

        inverted_setting =[
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        self.mobile = nn.Sequential(
            nn.Conv2d(3,32,2,1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace= True),
        )

        for t,c,n,s in inverted_setting:
            for i in range(n):
                if i == 0 :
                    self.mobile.append(InvertedResidual(input_channel,c,t,1, bias=False))
                else:
                    self.mobile.append(InvertedResidual(input_channel,c,t,s, bias=False))

            input_channel = c

        self.mobile.append(
            nn.Conv2d(input_channel,output_channel,1,1,0, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),
        )

        ## expansion, Projection layer 등은 InvertedResidual 만들어 진거고, 전체적인 layer 에서 생각하면 안된다.


        self.classifier = nn.Sequential(
            nn.Dropout(0.7),
            nn.Linear(output_channel, n_class),
        )


    def forward(self,x):
        x = self.mobile(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)

        return x


    def initialize(self):
        # ?? 이해 불가
