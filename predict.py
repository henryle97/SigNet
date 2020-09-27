import time

import torchvision
from PIL import ImageOps
import matplotlib.pyplot as plt
from pandas import np
from torchvision import transforms

from model.signet import load_model_from_checkpoint
from config.config_utils import get_config
from PIL import Image

import torch

class Predictor:
    def __init__(self, config):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.model = load_model_from_checkpoint(config['predict']['checkpoint'])
        self.model.to(self.device)
        self.model.eval()

        self.predict_transform = transforms.Compose([
            transforms.Resize((config['data']['height'], config['data']['width'])),
            ImageOps.invert,
            transforms.ToTensor(),
            # TODO: add normalize
        ])



    def predict(self, sig1, sig2):
        """

        :param sig1: PIL image model L
        :param sig2: PIL image model L
        :return:
        """
        sig1 = self.predict_transform(sig1)
        sig2 = self.predict_transform(sig2)
        sig1 = sig1.unsqueeze(0)
        sig2 = sig2.unsqueeze(0)
        sig1, sig2 = sig1.to(self.device), sig2.to(self.device)

        feature1, feature2 = self.model(sig1, sig2)
        distance = torch.pairwise_distance(feature1, feature2, 2)

        concatenated = torch.cat((sig1, sig2), 0)
        self.imshow(torchvision.utils.make_grid(concatenated),
               'Dissimilarity: {:.5f} Label: {}'.format(distance.item(), "label"))
        # print(distance)
        return distance

    def imshow(self, img, text=None, should_save=False):
        npimg = img.numpy()
        plt.axis("off")
        if text:
            plt.text(75, 8, text, style='italic', fontweight='bold',
                     bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


if __name__ == "__main__":

    model = torch.load

    config = get_config("./config/config.yml")
    predictor = Predictor(config)

    img1_path = "tests/hao.png"
    img2_path = "tests/hao3.png"

    img1 = Image.open(img1_path).convert('L')
    img2 = Image.open(img2_path).convert('L')

    t1 = time.time()
    euclid_distance = predictor.predict(img1, img2)
    euclid_distance = euclid_distance.item()



    print("Distance: ", euclid_distance)
    print("Total time: ", time.time() - t1)

