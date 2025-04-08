import os
import sys
from torch.autograd import Variable
from model_store import *
from utils.util import *

def getConfig(mode=1):
    store = [
        ['./weights/hynet_lib.pth', HyNet, 128],
        ['./weights/car_hynet.pth', CAR_HyNet, 128],
    ]
    if mode >= len(store):
        print('選擇的類型錯誤!')
        exit(-1)
    weights_path, model, dim = store[mode]
    return weights_path, model, dim


class NetFeature2D:
    def __init__(self, do_cuda=True, DLColor=True, mode=1, cuda="cuda:0"):
        self.G_weights_path, self.G_model, self.G_dim = getConfig(mode)
        print('mode[mode]:', mode)
        print('mode[G_weights_path]:', self.G_weights_path)
        print('mode[G_dim]:', self.G_dim)
        self.model_weights_path = self.G_weights_path
        self.model = self.G_model()
        print('model_weights_path:', self.model_weights_path)
        self.do_cuda = do_cuda & torch.cuda.is_available()
        print(f'cuda: {self.do_cuda} => {cuda}')
        self.device = torch.device(cuda if self.do_cuda else "cpu")
        # torch.set_grad_enabled(False)
        # Mag_factor是原始关键点比例被放大多少倍以从一个关键点生成一个patch
        self.mag_factor = 1.0
        # inference batch size
        self.batch_size = 512  # 512
        self.process_all = True  # 一次处理所有patches
        print('==> Loading pre-trained network.')
        self.checkpoint = torch.load(self.model_weights_path, map_location=self.device)
        self.model.load_state_dict(self.checkpoint)
        if self.do_cuda:
            self.model.to(self.device)
            print('Extracting on GPU')
        else:
            self.model = self.model.cpu()
            print('Extracting on CPU')
        self.model.eval()
        print('==> Successfully loaded pre-trained network.')

    def setConfig(self, mode=1):
        self.G_weights_path, self.G_model, self.G_dim = getConfig(mode)

    def getConfig(self):
        return self.G_weights_path, self.G_model, self.G_dim

    def compute_des_batches(self, patches, DLColor):
        dim = self.G_dim
        descriptors_for_net = np.zeros((len(patches), dim), dtype=np.float32)
        for i in range(0, len(patches), self.batch_size):
            data_a = patches[i: i + self.batch_size, :, :].astype(np.float32)
            if DLColor:
                data_a = torch.from_numpy(data_a).permute(0, 3, 1, 2)
            else:
                data_a = torch.from_numpy(data_a).unsqueeze(1)  # 3通道改了這裏!!!
            if self.do_cuda:
                data_a = data_a.to(self.device)
            data_a = Variable(data_a)
            with torch.no_grad():
                out_a = self.model(data_a)
                descriptors_for_net[i: i + self.batch_size] = out_a.cpu().detach().numpy().reshape(-1, dim)
        return descriptors_for_net

    def compute_des(self, patches):
        patches = torch.from_numpy(patches).float()
        patches = torch.unsqueeze(patches, 1)
        if self.do_cuda:
            patches = patches.cuda()
        with torch.no_grad():
            descrs = self.model(patches)
        return descrs.detach().cpu().numpy().reshape(-1, 128)

    def compute(self, patches, img, kps, mask=None):  # mask is a fake input
        num_kps = len(kps)
        des = []
        if num_kps > 0:
            # 通过向网络提供完整的patch张量来计算描述符
            # patches = extract_patches_array(img, kps, patch_size=32, mag_factor=self.mag_factor)
            # 缩小图像 用INTER_AREA更好，放大图像用 INTER_CUBIC更好；
            # patches = [cv2.resize(p, (64, 64), interpolation=cv2.INTER_CUBIC) for p in patches]
            patches = np.asarray(patches)
            patches = (patches / 255. - 0.443728476019) / 0.20197947209
            des = self.compute_des(patches)
        return kps, des

    def compute_sift(self, patches, kps, DLColor, mask=None):  # mask is a fake input
        des = []
        if len(kps) > 0: des = self.compute_des_batches(patches, DLColor).astype(np.float32)
        return kps, des
