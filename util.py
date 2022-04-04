import torch.nn as nn
import numpy as np
import torch

from PIL import Image
import torch.nn.functional as F

class BinOp():
    def __init__(self, model):
        # count the number of Conv2d and Linear
        count_targets = -1

        self.bin_range = []
        for m in model.named_modules():
            if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.Linear):
                count_targets += 1
                if 'branch' not in m[0] and count_targets != 0: # first layer and downsampling 1x1
                    self.bin_range.append(count_targets)
        del self.bin_range[-1] # last layer

        print(self.bin_range)

        self.num_of_params = len(self.bin_range)
        self.saved_params = []
        self.target_params = []
        self.target_modules = []
        self.gradient_scale_params = []
        self.gradient_bias_params = []
        index = -1
        for m in model.named_modules():
            if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.Linear):
                index = index + 1
                if index in self.bin_range:
                    tmp = m[1].weight.data.clone()
                    self.saved_params.append(tmp)
                    self.target_modules.append(m[1].weight)

    def binarization(self):
        self.clampConvParams()
        self.save_params()
        self.binarizeConvParams()

    def clampConvParams(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.clamp_(-1.0, 1.0)
        
    def save_params(self):
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)


    def binarizeConvParams(self): #original version
        m_tmp = 0
        for index in range(self.num_of_params):
            n = self.target_modules[index].data[0].nelement()
            s = self.target_modules[index].data.size()
            if len(s) == 4:
                if index%2==0:
                    m = (self.target_modules[index].data.norm(1, 3, keepdim=True) + self.target_modules[index+1].data.norm(1, 3, keepdim=True)) \
                                .sum(2, keepdim=True).sum(1, keepdim=True).div(2*n)     #(|x1|_1 + |x2|_1) / (2n)
                    m_tmp = m
                else:
                    m = m_tmp
            elif len(s) == 2:
                if index%2==0:
                    m = (self.target_modules[index].data.norm(1, 1, keepdim=True) + self.target_modules[index+1].data.norm(1, 1, keepdim=True)).div(2*n)
                    m_tmp = m
                else:
                    m = m_tmp
            self.target_modules[index].data.sign_().mul_(m.expand(s))

    def restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])

    def updateBinaryGradWeight(self):
        for index in range(self.num_of_params):
            weight = self.target_modules[index].data
            n = weight[0].nelement()
            s = weight.size()
            m_add = weight.sign().mul(self.target_modules[index].grad.data)
            if len(s) == 4:
                m_add = m_add.sum(3, keepdim=True)\
                        .sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            elif len(s) == 2:
                m_add = m_add.sum(1, keepdim=True).div(n).expand(s)
            m_add = m_add.mul(weight.sign())
            m = self.target_modules[index].grad.data
            #m[weight.lt(-1.)] = 0 ##+1.960
            #m[weight.gt(+1.)] = 0
            index_tmp = weight.lt(-1.960) + weight.gt(+1.960)            
            m[index_tmp] = 0            
            self.target_modules[index].grad.data = m.add(m_add)

def TS_Loss(teacher_output, student_output, opt):
        """The KL-Divergence loss for the model and refined labels output.
        output must be a pair of (model_output, refined_labels), both NxC tensors.
        The rows of refined_labels must all add up to one (probability scores);
        however, model_output must be the pre-softmax output of the network."""
        if teacher_output.requires_grad:
            raise ValueError("Teacher output should not require gradients.")

        """added for teacher_output must all add up to one"""
        teacher_output = F.softmax(teacher_output, dim=1)

        student_output_log_prob = F.log_softmax(student_output, dim=1)

        # Loss is -dot(model_output_log_prob, teacher_output). Prepare tensors
        # for batch matrix multiplicatio
        teacher_output = teacher_output.unsqueeze(1)

        student_output_log_prob = student_output_log_prob.unsqueeze(2)

        # Compute the loss, and average/sum for the batch.
        cross_entropy_loss = -torch.bmm(teacher_output, student_output_log_prob)
        if opt=='mean':
            cross_entropy_loss = cross_entropy_loss.mean()
        elif opt=='sum':
            cross_entropy_loss = cross_entropy_loss.sum()

        if isinstance(cross_entropy_loss, tuple):
            loss_value, outputs = cross_entropy_loss
        else:
            loss_value = cross_entropy_loss

        return loss_value