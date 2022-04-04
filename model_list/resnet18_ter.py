import torch
import os
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

import TerActivateFunc_cuda

from util import *

__all__ = ['resnet18_ter']

class SymmTerActiveF(torch.autograd.Function):
    '''
    Ternarize the input activations to 0, -1, +1
    '''
    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        delta = 1.
        input = delta/2. * ((input-delta/2.).sign() + (input+delta/2.).sign())
        return input

    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        index = input.ge(1.0) + input.le(-1.0)
        grad_input[index] = 0.
        return grad_input

class SymmTerActive(nn.Module):
    def __init__(self):
        super(SymmTerActive, self).__init__()

    def forward(self, x):
        x = SymmTerActiveF.apply(x)
        return x

#Ter Activation CUDA
class TerSymmActiveF_cuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        ctx.backend = TerActivateFunc_cuda
        output = torch.zeros_like(input)
        output = ctx.backend.forward(input, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        ctx.backend.backward(ctx.input, grad_input, 0.)
        return grad_input

class SymmTerActive_cuda(nn.Module):
    def __init__(self):
        super(SymmTerActive_cuda, self).__init__()
        self.ter_act = TerSymmActiveF_cuda.apply

    def forward(self, x):
        out = self.ter_act(x)
        return out



class BasicBlock(nn.Module):
    def __init__(self, input_channels, output_channels, 
            stride = -1, has_branch = True, ActiveFunction=None):
        super(BasicBlock, self).__init__()
        self.has_branch = has_branch

        self.bn1 = nn.BatchNorm2d(input_channels)
        self.ba1 = ActiveFunction()
        self.prelu1 = nn.PReLU(output_channels)
        #self.conv1_1 = nn.Conv2d(input_channels, output_channels,
        #                    kernel_size=3, stride=stride, padding=1,
        #                    )
        #self.conv1_2 = nn.Conv2d(input_channels, output_channels,
        #                    kernel_size=3, stride=stride, padding=1,
        #                    )
        self.conv1 =  nn.Conv2d(input_channels, output_channels,
                            kernel_size=3, stride=stride, padding=1,
                            )

        self.bn_add1 = nn.BatchNorm2d(output_channels)

        self.bn2 = nn.BatchNorm2d(output_channels)
        self.ba2 = ActiveFunction()
        self.prelu2 = nn.PReLU(output_channels)
        #self.conv2_1 = nn.Conv2d(output_channels, output_channels,
        #                    kernel_size=3, stride=1, padding=1,
        #                    )
        #self.conv2_2 = nn.Conv2d(output_channels, output_channels,
        #                    kernel_size=3, stride=1, padding=1,
        #                    )
        self.conv2 = nn.Conv2d(output_channels, output_channels,
                            kernel_size=3, stride=1, padding=1,
                            )

        self.bn_add2 = nn.BatchNorm2d(output_channels)

        if has_branch:
            self.branch = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

    def forward(self, x):
        # conv block
        out = self.bn1(x)
        out = self.ba1(out)
        ###### short-cut
        if self.has_branch:
            short_cut = self.branch(x)
            short_cut = torch.cat([short_cut, short_cut.mul(0.)], 1)
        else:
            short_cut = x
        ###### short-cut
        #out = self.conv1_1(out) + self.conv1_2(out)
        out = self.conv1(out)

        out = self.bn_add1(out)
        out += short_cut
        out = self.prelu1(out)
        add = out

        out = self.bn2(out)
        out = self.ba2(out)
        #out = self.conv2_1(out) + self.conv2_2(out)
        out = self.conv2(out)
        out = self.bn_add2(out)
        out += add
        out = self.prelu2(out)
        return out


class ResNet18_BNN(nn.Module):
    def __init__(self, num_classes=1000, ActiveFunction=None):
        super(ResNet18_BNN, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            BasicBlock(64, 64, stride = 1, has_branch = False, ActiveFunction=ActiveFunction),
            BasicBlock(64, 64, stride = 1, has_branch = False, ActiveFunction=ActiveFunction),

            BasicBlock(64, 128, stride = 2, has_branch = True, ActiveFunction=ActiveFunction),
            BasicBlock(128, 128, stride = 1, has_branch = False, ActiveFunction=ActiveFunction),
            
            BasicBlock(128, 256, stride = 2, has_branch = True, ActiveFunction=ActiveFunction),
            BasicBlock(256, 256, stride = 1, has_branch = False, ActiveFunction=ActiveFunction),

            BasicBlock(256, 512, stride = 2, has_branch = True, ActiveFunction=ActiveFunction),
            BasicBlock(512, 512, stride = 1, has_branch = False, ActiveFunction=ActiveFunction)
        )
        self.avgpool = nn.AvgPool2d(7, stride = 1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x


ActiveFunction_dict = {
    ###float###
    'hard_tanh': nn.Hardtanh,
    'relu': nn.ReLU,
    "SymmTerActive_cuda": SymmTerActive_cuda,
    "SymmTerActive": SymmTerActive,

}

def resnet18_ter(pretrained=False, activeF = 'SymmTerActive_cuda'):
    model = ResNet18_BNN(ActiveFunction=ActiveFunction_dict[activeF])
    return model
