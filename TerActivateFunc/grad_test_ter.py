import torch
import TerActivateFunc_cuda

class TerActivateFunc(torch.autograd.Function):
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
        ctx.backend.backward(ctx.input, grad_input, -0.3)
        return grad_input

def naive_forward(x):
    delta = 0.5
    return delta*((x+delta).sign() + (x-delta).sign())

def naive_backward(input, grad_output):
    delta = 0.5
    #grad_output[input<0] = 0
    #grad_output[input>4*delta] = 0
    #return grad_output

    grad_output[input<-2.*delta] = 0
    grad_output[input>2.*delta] = 0
    return grad_output

def main():
    cuda0 = torch.device('cuda:0')
    A = TerActivateFunc.apply

    #check for forward
    input = torch.randn(4, 4, 1, requires_grad=True, device=cuda0)
    output = A(input)
    print(output)
    print(torch.equal(output, naive_forward(input)))

    #check for backward
    grad_output = torch.randn(4, 4, 1).cuda()
    output.backward(grad_output)
    print(torch.equal(naive_backward(input, grad_output), input.grad))

if __name__ == '__main__':
    torch.manual_seed(618)
    main()
