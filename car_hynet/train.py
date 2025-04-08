import ast
import argparse
from matplotlib import pyplot as plt
import torch.optim as optim
import os
import torch
import time
from model_store import *
from utils.util import *
import wandb
import matplotlib
# matplotlib.use('TkAgg')  # 使用 TkAgg 后端
# wandb.init(project="CAR-HyNet")

def draw_batch(batch, show=False):
    fig, axs = plt.subplots(8, 8, figsize=(10, 10))
    for i in range(8):
        for j in range(8):
            # 计算当前子图的索引
            index = i * 8 + j
            # 将数据从[0, 1]归一化到[0, 255]
            img = (batch[index] * 255).cpu().numpy().astype(np.uint8)
            # 因为matplotlib默认情况下期望图像为格式(W, H, C)，我们需要将图像数据转置
            img = np.transpose(img, (1, 2, 0))
            axs[i, j].imshow(img)
            axs[i, j].axis('off')  # 不显示坐标轴
    # 调整子图间距
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    if show: plt.show()

def train_net(desc_name, nb_batch_per_epoch):
    net.train()
    running_loss = 0.0
    running_dist_pos = 0.0
    running_dist_neg = 0.0
    bar = tqdm(range(nb_batch_per_epoch), ncols=90)
    for batch_loop in bar:
        index_batch = index_train[epoch_loop][batch_loop]
        batch = patch_train[index_batch]
        batch = batch.to(torch.float32)

        if flag_dataAug:
            batch = data_aug(batch, num_pt_per_batch)

        draw_batch(batch[0::2], show=False)
        draw_batch(batch[1::2], show=True)

        batch = batch.to(device)
        desc_L, desc_raw_L = net(batch[0::2], mode='train')
        desc_R, desc_raw_R = net(batch[1::2], mode='train')
        loss, dist_pos, dist_neg = loss_desc.compute(desc_L, desc_R, desc_raw_L, desc_raw_R)

        running_loss = running_loss + loss.item()
        running_dist_pos += dist_pos.item()
        running_dist_neg += dist_neg.item()
        bar.set_description('[Epoch {}: {}/{}]pos: {:.4f}|neg: {:.4f}|loss: {:.4f}'.format(
            epoch_loop+1,
            batch_loop + 1,
                nb_batch_per_epoch,
            running_dist_pos / (batch_loop + 1),
            running_dist_neg / (batch_loop + 1),
            running_loss / (batch_loop + 1)))
        if (batch_loop+1)%200==0 or (batch_loop+1)==len(bar):
            log_file.write('[Epoch {}: {}/{}]pos: {:.4f}|neg: {:.4f}|loss: {:.4f}\n'.format(
                epoch_loop+1,
                batch_loop + 1,
                nb_batch_per_epoch,
                running_dist_pos / (batch_loop + 1),
                running_dist_neg / (batch_loop + 1),
                running_loss / (batch_loop + 1)))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # wandb.log({'dist_pos': running_dist_pos / (batch_loop + 1), 'dist_neg': running_dist_neg / (batch_loop + 1), 'loss': loss})
    return

def test_net(device, net, patch, pointID, index, dim_desc=128, sz_batch=500):
    net.eval()
    nb_patch = pointID.size
    nb_loop = int(np.ceil(nb_patch/sz_batch))
    desc = torch.zeros(nb_patch, dim_desc)
    with torch.set_grad_enabled(False):
        for i in range(nb_loop):
            st = i * sz_batch
            en = np.min([(i + 1) * sz_batch, nb_patch])
            batch = patch[st:en].to(device)
            out_desc = net(batch, mode='eval')
            out_desc = out_desc.to('cpu')
            desc[st:en] = out_desc
            print(': {} of {}'.format(i, nb_loop), end='\r')

    fpr95 = cal_fpr95(desc, pointID, index)
    return fpr95

parser = argparse.ArgumentParser(description='pyTorch descNet')
parser.add_argument('--data_root', type=str, default=r'../datasets')  # path containing the UBC and HPatches data set
parser.add_argument('--network_root', type=str, default='weights')  # path containing the trained models
parser.add_argument('--train_set', type=str, default='liberty')  # notredame, liberty, yosemite, hpatches,
parser.add_argument('--train_split', type=str, default='all')  # full
parser.add_argument('--suffix', type=str, default='')
parser.add_argument('--sz_patch', type=int, default=32)
parser.add_argument('--num_pt_per_batch', type=int, default=64)
parser.add_argument('--dim_desc', type=int, default=128)
parser.add_argument('--nb_pat_per_pt', type=int, default=2)
parser.add_argument('--epoch_max', type=int, default=1)
parser.add_argument('--margin', type=float, default=1.2)  #1.0
parser.add_argument('--flag_dataAug', type=ast.literal_eval, default=True)
parser.add_argument('--is_sosr', type=ast.literal_eval, default=False)
parser.add_argument('--knn_sos', type=int, default=8)
parser.add_argument('--optim_method', type=str, default='Adam')
parser.add_argument('--lr_scheduler', type=str, default='None')  # CosineAnnealing 余弦退火学习率
parser.add_argument('--desc_name', type=str, default='HyNet')
parser.add_argument('--alpha', type=float, default=2)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--drop_rate', type=float, default=0.1)  # 0.3
args = parser.parse_args()


seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
data_root = args.data_root
train_set = args.train_set
sz_patch = args.sz_patch
epoch_max = args.epoch_max
num_pt_per_batch = args.num_pt_per_batch
nb_pat_per_pt = args.nb_pat_per_pt
num_pt_per_batch = args.num_pt_per_batch
dim_desc = args.dim_desc
margin = args.margin
drop_rate = args.drop_rate
is_sosr = args.is_sosr
knn_sos = args.knn_sos
flag_dataAug = args.flag_dataAug
optim_method = args.optim_method
lr_scheduler = args.lr_scheduler
alpha = args.alpha
desc_name = args.desc_name
train_split = args.train_split
lr = args.lr

#-----------------------------获取保存文件夹名称----------------------------------------#
folder_name = desc_name + '_' + train_set
if train_set == 'hpatches': folder_name += '_split_' + train_split
folder_name += '_sz_' + str(sz_patch)
folder_name += '_pt_' + str(num_pt_per_batch)
folder_name += '_pat_' + str(nb_pat_per_pt)
folder_name += '_dim_' + str(dim_desc)
if args.desc_name == 'HyNet': folder_name += '_alpha_' + str(alpha)
folder_name += '_margin_' + str(margin)
folder_name += '_drop_' + str(drop_rate)
folder_name += '_lr_' + str(lr)
folder_name += '_' + optim_method + '_' + lr_scheduler
if flag_dataAug: folder_name += '_aug'
if len(args.suffix) > 0: folder_name += '-' + args.suffix  # for debugging
folder_name += '-' + str(int(time.time()))
net_dir = os.path.join(args.network_root, 'network', folder_name)
print(net_dir)
if not os.path.exists(net_dir): os.makedirs(net_dir)

#-------------------------------文件夹准备完毕--------------------------------------#
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = CAR_HyNet(dim_desc=dim_desc, drop_rate=drop_rate)
net.to(device)

#-------------------------------数据预处理--------------------------------------#
ubc_subset = ['yosemite', 'liberty', 'notredame']
if train_set == 'liberty' or args.train_set == 'notredame' or args.train_set == 'yosemite':
    patch_train, pointID_train, index_train = load_UBC_for_train(data_root, train_set,
                                                                 sz_patch,
                                                                 num_pt_per_batch, nb_pat_per_pt,
                                                                 epoch_max, color=True)
    test_set = []
    for val in ubc_subset:
        if val != train_set: test_set.append(val)
elif train_set == 'hpatches':
    if args.train_split == 'all':
        patch_train, pointID_train, index_train = load_hpatches_for_train(args.data_root,
                                                                          args.sz_patch,
                                                                          args.num_pt_per_batch,
                                                                          args.nb_pat_per_pt,
                                                                          args.epoch_max)
    else:
        patch_train, pointID_train, index_train = load_hpatches_split_train(data_root,
                                                                            sz_patch,
                                                                            num_pt_per_batch,
                                                                            nb_pat_per_pt,
                                                                            epoch_max,
                                                                            split_name=train_split)
    test_set = ['yosemite', 'liberty', 'notredame']
nb_batch_per_epoch = len(index_train[0])  # Each epoch has equal number of batches

patch_test = {}
pointID_test = {}
index_test = {}
for i, val in enumerate(test_set):
    patch_test[val], pointID_test[val], index_test[val] = load_UBC_for_test(args.data_root, val, args.sz_patch, color=True)
    patch_test[val] = torch.from_numpy(patch_test[val])
    patch_test[val] = patch_test[val].to(torch.float32)
    index_test[val] = index_test[val]

if optim_method == 'Adam':
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
elif optim_method == 'SGD':
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9)

if lr_scheduler == 'CosineAnnealing':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epoch_max, eta_min=1e-7, last_epoch=-1)

# file names
file_fpr95 = 'fpr95_'
for i, val in enumerate(test_set): file_fpr95 = file_fpr95 + val + '_'
file_fpr95_best = file_fpr95[0:-1] + '_best.npy'
file_fpr95 = file_fpr95[0:-1] + '.npy'
file_fpr95 = os.path.join(net_dir, file_fpr95)
file_fpr95_best = os.path.join(net_dir, file_fpr95_best)
net_best_name = os.path.join(net_dir, 'net-best.pth')
#log_file = os.path.join(net_dir, 'log.txt')
log_file = open('log.txt', 'w+')

# descriptor type
loss_desc = Loss_HyNet(device, num_pt_per_batch, dim_desc, margin, alpha, is_sosr, knn_sos)

# 开始训练
# file names
desc_dir = os.path.join(net_dir, 'desc')
if not os.path.exists(desc_dir): os.makedirs(desc_dir)

fpr95 = []
best_mAP = 0
fpr95_best_per_test = {}
net_best_params = ''
for epoch_loop in range(args.epoch_max):
    # train
    train_net(desc_name, nb_batch_per_epoch)
    if lr_scheduler != 'None': scheduler.step()
    net_name = os.path.join(net_dir, 'net-epoch-{}.pth'.format(epoch_loop + 1))
    torch.save(net.state_dict(), net_name)
    # validation
    # fpr95 as target
    fpr95_per_epoch = []
    # 对每个test_set中的数据集进行验证
    for i, val in enumerate(test_set):
        print('validation:', val)
        fpr95_per_epoch.append(test_net(device, net, patch_test[val], pointID_test[val], index_test[val], args.dim_desc))
    if len(fpr95_per_epoch) > 0:
        fpr95.append(fpr95_per_epoch)
        np.save(file_fpr95, fpr95)
        #fpr_avg = np.mean(np.array(fpr95_per_epoch))
        fpr_avg = 0
        for val, fpr in zip(test_set, fpr95_per_epoch):
            if val != train_set: fpr_avg += fpr
        fpr_avg /= 2
        if epoch_loop == 0:
            fpr_avg_best = fpr_avg
            epoch_best = 0
        else:
            if fpr_avg_best > fpr_avg:
                fpr_avg_best = fpr_avg
                fpr_best = fpr95_per_epoch.copy()
                fpr_best.append(epoch_loop+1)
                torch.save(net.state_dict(), net_best_name)
                np.save(file_fpr95_best, fpr_best)
        for t, fpr in zip(test_set, fpr95_per_epoch):
            print('{0} : {1:.4f}%'.format(t, fpr*100))
            log_file.write('{0} : {1:.4f}%\n'.format(t, fpr*100))

            value = fpr95_best_per_test.get(t)
            if value: fpr95_best_per_test[t] = value if value < fpr*100 else fpr*100
            else: fpr95_best_per_test[t] = fpr*100
        print('fpr_avg: {}; fpr_avg_best: {}'.format(fpr_avg*100, fpr_avg_best*100))
        print(fpr95_best_per_test.items())
        log_file.write(str(fpr95_best_per_test.items()) + '\n')
        log_file.write('Best : {0:.4f}%\n'.format(fpr_avg_best*100))
        log_file.flush()
        # wandb.log(dict(fpr95_best_per_test.items()))
        # wandb.log({'fpr_avg': fpr_avg*100, 'fpr_avg_best': fpr_avg_best*100})


print('all done!')
print(folder_name)
print(net_best_name)




