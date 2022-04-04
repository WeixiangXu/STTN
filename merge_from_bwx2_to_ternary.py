import torch
import model_list
import util

import os

def save_checkpoint(state, path='./', filename='checkpoint.pth.tar'):
    saved_path = os.path.join(path, filename)
    torch.save(state, saved_path)


#resnet18
float_model = model_list.resnet18_bwx2()
float_model = torch.nn.DataParallel(float_model)
float_checkpoint = torch.load('./checkpoint_ter/res18_bwx2_ta/model_best.pth.tar')
save_path = './ternary'
filename = 'resnet18_ter.pth.tar'

float_model.load_state_dict(float_checkpoint['state_dict'])
bin_op = util.BinOp(float_model)
bin_op.binarization()

#
ter_model = model_list.resnet18_ter()
ter_model = torch.nn.DataParallel(ter_model)

#save
ter_model_dict = ter_model.state_dict()
float_dict = {k: v for k, v in float_checkpoint['state_dict'].items() if k in ter_model_dict}
ter_model_dict.update(float_dict)
ter_model.load_state_dict(ter_model_dict)

for index in range(4, 12):
  ter_model.module.features[index].conv1.weight.data = float_model.module.features[index].conv1_1.weight.data + float_model.module.features[index].conv1_2.weight.data 
  ter_model.module.features[index].conv2.weight.data = float_model.module.features[index].conv2_1.weight.data + float_model.module.features[index].conv2_2.weight.data
  ter_model.module.features[index].conv1.bias.data = float_model.module.features[index].conv1_1.bias.data + float_model.module.features[index].conv1_2.bias.data
  ter_model.module.features[index].conv2.bias.data = float_model.module.features[index].conv2_1.bias.data + float_model.module.features[index].conv2_2.bias.data

save_checkpoint({
   'state_dict': ter_model.state_dict(),
}, path=save_path, filename=filename)

print("Successfully change 2 parallel binary weights into a ternary weights.")