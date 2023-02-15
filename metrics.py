import os
import torch.utils.data
from torch.nn import DataParallel
from datetime import datetime
from torch.optim.lr_scheduler import MultiStepLR
from config import BATCH_SIZE, PROPOSAL_NUM, SAVE_FREQ, LR, WD, resume, save_dir
from core import model, dataset
from core.utils import init_log, progress_bar

os.environ['CUDA_VISIBLE_DEVICES'] = '1' #'0,1,2,3'
start_epoch = 0
save_dir = os.path.join(save_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
if os.path.exists(save_dir):
    raise NameError('model dir exists!')
os.makedirs(save_dir)
logging = init_log(save_dir)
_print = logging.info


print('testset load.....')
testset = dataset.CUB(root='./CUB_200_2011', is_train=False, data_len=None)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=8, drop_last=False)

# define model
net = model.Concat_net(topN=PROPOSAL_NUM)
if True:
    ckpt = torch.load('./K4CU903.ckpt')
    net.load_state_dict(ckpt['net_state_dict'])
    start_epoch = ckpt['epoch'] + 1
creterion = torch.nn.CrossEntropyLoss()

# define optimizers
raw_parameters = list(net.pretrained_model.parameters())
part_parameters = list(net.proposal_net.parameters())
concat_parameters = list(net.concat_net.parameters())
partcls_parameters = list(net.partcls_net.parameters())

raw_optimizer = torch.optim.SGD(raw_parameters, lr=LR, momentum=0.9, weight_decay=WD)
concat_optimizer = torch.optim.SGD(concat_parameters, lr=LR, momentum=0.9, weight_decay=WD)
part_optimizer = torch.optim.SGD(part_parameters, lr=LR, momentum=0.9, weight_decay=WD)
partcls_optimizer = torch.optim.SGD(partcls_parameters, lr=LR, momentum=0.9, weight_decay=WD)
schedulers = [MultiStepLR(raw_optimizer, milestones=[120, 200], gamma=0.1),
              MultiStepLR(concat_optimizer, milestones=[120, 200], gamma=0.1),
              MultiStepLR(part_optimizer, milestones=[120, 200], gamma=0.1),
              MultiStepLR(partcls_optimizer, milestones=[120, 200], gamma=0.1)]
net = net.cuda()
net = DataParallel(net)

	# evaluate on test set
test_loss = 0
test_correct = 0
total = 0
net.eval()
confuse_matrix = [[0 for j in range(0, 88)] for i in range(0, 88)]
for i, data in enumerate(testloader):
    with torch.no_grad():
        img, label = data[0].cuda(), data[1].cuda()
        batch_size = img.size(0)
        _, concat_logits, _, _, _ = net(img)
        print('concat_logits:',concat_logits.shape)
        # calculate loss
        concat_loss = creterion(concat_logits, label)
        # calculate accuracy
        _, concat_predict = torch.max(concat_logits, 1)
        total += batch_size
        print('lable' ,label.data)      # shape = batch_size
        print('concat_predict.data:',concat_predict.data)       # shape = batch_size
        for i in range(0,len(label.data)):
            confuse_matrix[label.data[i]][concat_predict.data[i]] += 1
        test_correct += torch.sum(concat_predict.data == label.data)
        test_loss += concat_loss.item() * batch_size
        progress_bar(i, len(testloader), 'eval test set')

test_acc = float(test_correct) / total
test_loss = test_loss / total
# _print(
#     'epoch:{} - test loss: {:.3f} and test acc: {:.3f} total sample: {}'.format(
#         1,
#         test_loss,
#         test_acc,
#         total))
print(confuse_matrix)
print("ACC:",test_acc)
arrs= confuse_matrix
ARRS = []
f=open('confuse_matrix_k5cu903.txt','w+')
for i in range(88):
 jointsFrame = arrs[i] #每行
 ARRS.append(jointsFrame)
 for Ji in range(88):
  strNum = str(jointsFrame[Ji])
  f.write(strNum)
  f.write(' ')
 f.write('\n')
f.close()

print('finishing test')
