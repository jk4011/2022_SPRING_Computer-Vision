from model import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
import glob
import pandas
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

device = 'cuda:0'


class FaceDataset(Dataset):
    def __init__(self, root, transforms_=None, img_size=128, mask_size=64, method="train"):
        self.transform = transforms.Compose(transforms_)
        self.img_size = img_size
        self.mask_size = mask_size
        self.mode = method
        self.root = root
        self.files = sorted(glob.glob("%s/*.jpg" % root))
        self.files = self.files[:-4000] if self.mode == "train" else self.files[-4000:]

    def apply_random_mask(self, img):
        """Randomly masks image"""
        y1, x1 = np.random.randint(0, self.img_size - self.mask_size, 2)
        y2, x2 = y1 + self.mask_size, x1 + self.mask_size
        masked_part = img[:, y1:y2, x1:x2]
        masked_img = img.clone()
        masked_img[:, y1:y2, x1:x2] = 1
        return masked_img, masked_part

    def apply_center_mask(self, img):
        """Mask center part of image"""
        # Get upper-left pixel coordinate
        i = (self.img_size - self.mask_size) // 2
        masked_img = img.clone()
        masked_img[:, i : i + self.mask_size, i : i + self.mask_size] = 1
        return masked_img, i

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img = self.transform(img)
        if self.mode == "train":
            # For training data perform random mask
            masked_img, aux = self.apply_random_mask(img)
        else:
            # For test data mask the center of the image
            masked_img, aux = self.apply_center_mask(img)
        return img, masked_img, aux

    def __len__(self):
        return len(self.files)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

class Trainer(object):
    def __init__(self, epochs, batch_size, lr):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = lr
        self._build_model()
        self.binary_cross_entropy = torch.nn.BCELoss()
        self.p_loss = torch.nn.L1Loss()
        transforms_ = [transforms.Resize((128, 128), Image.BICUBIC), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        dataset = FaceDataset(root='../data/train', method='train', transforms_=transforms_) #Change this path
        self.root = dataset.root

        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.optimizer_G = torch.optim.Adam(self.gnet.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.dnet.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))

        print("Training...")

    def _build_model(self):
        gnet, dnet = Generator(), Discriminator()
        self.gnet = gnet.to(device)
        self.dnet = dnet.to(device)
        self.gnet.apply(weights_init_normal)
        self.dnet.apply(weights_init_normal)
        self.gnet.train()
        self.dnet.train()

        print('Finish build model.')

    def train(self):
        date = '20211017'
        for epoch in tqdm.tqdm(range(self.epochs + 1)):
            if epoch % 10 == 0:
                torch.save(self.gnet.state_dict(), f'./model_gan_lamb_{str(epoch)}.pth') #Change this path

            for batch_idx, (imgs, masked_imgs, masked_parts) in enumerate(self.dataloader):
                Tensor = torch.cuda.FloatTensor
                patch_h, patch_w = int(64 / 2 ** 3), int(64 / 2 ** 3)
                patch = (1, patch_h, patch_w)

                valid = Variable(Tensor(imgs.shape[0], *patch).fill_(1.0), requires_grad=False)
                fake = Variable(Tensor(imgs.shape[0], *patch).fill_(0.0), requires_grad=False)

                imgs = Variable(imgs.type(Tensor))
                masked_imgs = Variable(masked_imgs.type(Tensor))
                masked_parts = Variable(masked_parts.type(Tensor))


                # Generater loss
                # minimize - log(1-D(G(z)))
                g_output = self.gnet(masked_imgs)
                d_output_fake = self.dnet(masked_parts)

                self.optimizer_G.zero_grad() 

                l1_loss = self.p_loss(g_output, masked_parts)

                g_loss = - self.binary_cross_entropy(valid, 1 - d_output_fake)

                lamb = 0.5
                loss = g_loss * lamb + l1_loss #
                loss.backward()

                self.optimizer_G.step()


                # Discriminator loss
                # minimize - log(D(x)) + log(1-D(G(z)))
                g_output = self.gnet(masked_imgs)
                d_output_real = self.dnet(g_output)
                d_output_fake = self.dnet(masked_parts)
                
                self.optimizer_D.zero_grad()

                d_loss_real = - self.binary_cross_entropy(valid, d_output_real)
                d_loss_real.backward()

                d_loss_fake = self.binary_cross_entropy(fake, d_output_fake)
                d_loss_fake.backward()

                self.optimizer_D.step()


                # print("[Epoch %d/%d] [Batch %d/%d]" % (epoch, self.epochs+1, batch_idx, len(self.dataloader)))


class Tester(object):
    def __init__(self, batch_size):
        self._build_model()
        transforms_ = [transforms.Resize((128, 128), Image.BICUBIC), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        dataset = FaceDataset(root='../data/test', method='test', transforms_=transforms_)
        self.root = dataset.root
        self.test_dataloader = DataLoader(dataset, batch_size=6, shuffle=False)

        print("Testing...")

    def _build_model(self):
        gnet = Generator()
        self.gnet = gnet.to(device)
        self.gnet.load_state_dict(torch.load('./model_gan_lamb_50.pth')) #Change this path
        self.gnet.eval()
        print('Finish build model.')

    def test(self):
        Tensor = torch.cuda.FloatTensor
        samples, masked_samples, i = next(iter(self.test_dataloader))
        samples = Variable(samples.type(Tensor))
        masked_samples = Variable(masked_samples.type(Tensor))
        i = i[0].item()  # Upper-left coordinate of mask

        # Generate inpainted image
        gen_mask = self.gnet(masked_samples)
        filled_samples = masked_samples.clone()
        filled_samples[:, :, i : i + 64, i : i + 64] = gen_mask

        # Save sample
        sample = torch.cat((masked_samples.data, filled_samples.data, samples.data), -2)
        save_image(sample, "../test.png", nrow=6, normalize=True)   #Change this path

def main():

    epochs = 100
    batchSize = 64
    learningRate = 0.0002

    # trainer = Trainer(epochs, batchSize, learningRate)
    # trainer.train()

    tester = Tester(batchSize)
    tester.test()

if __name__ == '__main__':
    main()
