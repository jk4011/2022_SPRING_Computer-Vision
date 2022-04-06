from model import VAE
import torch
import numpy as np
import torchvision
import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
import torch.nn.functional as F
from glob import glob
import pandas
# from google.colab.patches import cv2_imshow
import utils

device = 'cuda:0'

class MVDataset(Dataset):
    def __init__(self, method=None, data_name='bottle', folder_name="good"):
        self.data_name = data_name
        self.root = f'../data/{self.data_name}/' #Change this path
        self.x_data = []
        self.y_data = []

        if method == 'train':
            self.root = self.root + 'train/good/'
            self.img_path = sorted(glob(self.root + '*.png'))
 
        elif method == 'test':
            self.root = self.root + 'test/' + folder_name + '/'
            self.img_path = sorted(glob(self.root + '*.png'))

        print("Load data")
        for i in tqdm.tqdm(range(len(self.img_path))):
            img = cv2.imread(self.img_path[i], cv2.IMREAD_COLOR)
            # print(self.img_path[i])
            b, g, r = cv2.split(img)
            img = cv2.merge([r, g, b])
            img = cv2.resize(img, dsize=(256, 256))
            #cv2.imwrite('test_%d.png' % i, img)

            self.x_data.append(img)
            self.y_data.append(img)

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        transform1 = torchvision.transforms.ToTensor()
        new_x_data = transform1(self.x_data[idx])
        return new_x_data, self.y_data[idx]


class Trainer(object):
    def __init__(self, data_name, epochs, batch_size, lr):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = lr
        self._build_model()
        self.binary_cross_entropy = torch.nn.BCELoss()

        self.data_name = data_name
        dataset = MVDataset(method='train', data_name=data_name)
        self.root = dataset.root
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = self.learning_rate, betas=(0.9, 0.999))

        # Load of pretrained_weight file
        print("Training...")

    def _build_model(self):
        net = VAE()
        self.net = net.to(device)
        self.net.train()

        print('Finish build model.')

    def vae_loss(self, recon_x, x, mu, logvar):
        recon_loss = self.binary_cross_entropy(recon_x.view(-1, 256*256*3), x.view(-1, 256*256*3))
        kldivergence = -0.5 * torch.sum(1+ logvar - mu.pow(2) - logvar.exp()) 
        return recon_loss + 0.000001 * kldivergence

    def train(self):
        date = '20220331'
        for epoch in tqdm.tqdm(range(self.epochs + 1)):
            if epoch == 10 or epoch == 500:
                torch.save(self.net.state_dict(), "_".join(['../model/model', self.data_name,  str(epoch) + "epoch.pth"])) #Change this path

            total_loss = 0

            for batch_idx, samples in enumerate(self.dataloader):
                x_train, y_train = samples
                x_train, y_train = x_train.to(device), y_train.to(device)
        
                # todo : implement the training code
                recon_x, latent_mu, latent_logvar = self.net(x_train)
                loss = self.vae_loss(recon_x, x_train, latent_mu, latent_logvar)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss
                
            print(f"-------------------------------------")
            print(f"epoch : {epoch}")
            print(f"loss  : {total_loss}")
            

        print('Finish training.')


class Tester(object):
    def __init__(self, batch_size, model_name, data_name='bottle', folder_name="broken"):
        self.batch_size = batch_size
        self._build_model()

        self.folder_name = folder_name
        dataset = MVDataset(method='test', data_name=data_name, folder_name=folder_name)
        self.root = dataset.root
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        self.datalen = dataset.__len__()
        self.mse_all_img = []

        # Load of pretrained_weight file
        self.model_name = model_name
        weight_PATH = f'../model/{self.model_name}.pth' #Change this path
        self.net.load_state_dict(torch.load(weight_PATH))

        print("Testing...")

    def _build_model(self):
        net = VAE()
        self.net = net.to(device)
        self.net.eval()

        print('Finish model initialization.')

    def test(self):
        for batch_idx, samples in enumerate(self.dataloader):
            x_test, y_test = samples
            out = self.net(x_test.cuda())

            x_test2 = 256. * x_test
            out2 = 256. * out[0]

            abnomal = utils.compare_images_colab(x_test2[0].clone().permute(1, 2, 0).cpu().detach().numpy(), out2[0].clone().permute(1, 2, 0).cpu().detach().numpy(), None, 0.2)   

            directory = f"../{self.model_name}_output/"
            utils.mkdirs(directory)
            cv2.imwrite(directory + f'{self.folder_name}_test_{batch_idx}_ori.png', cv2.cvtColor(x_test2[0].clone().permute(1, 2, 0).cpu().detach().numpy(), cv2.COLOR_RGB2BGR))
            cv2.imwrite(directory + f'{self.folder_name}_test_{batch_idx}_gen.png', cv2.cvtColor(out2[0].clone().permute(1, 2, 0).cpu().detach().numpy(), cv2.COLOR_RGB2BGR))
            cv2.imwrite(directory + f'{self.folder_name}_test_{batch_idx}_diff.png', abnomal)

def main():

    epochs = 500
    batchSize = 1
    learningRate = 1e-4

    trainer = Trainer("bottle", epochs, batchSize, learningRate)
    trainer.train()

    tester = Tester(batchSize, "model_toothbrush_500epoch0.00001", "toothbrush", "defective")
    tester.test()

if __name__ == '__main__':
    main()
