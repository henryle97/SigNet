from model.signet import SigNet
from model.loss import ContrastiveLoss
import os
from dataset.signet_dataset import get_data_loader
from PIL import ImageOps
import torch
import torch.optim as optim
from torchvision import transforms
from utils.metric import accuracy
from config.config_utils import get_config

from utils.logger import Logger
class Trainer():
    def __init__(self, config_path):
        config = get_config(config_path)
        print(config)

        if torch.cuda.is_available():
            self.device = torch.device(config['device'])
        else:
            self.device = torch.device('cpu')
        print("Device: ", self.device)

        seed = config['seed']
        alpha = config['loss']['alpha']
        beta = config['loss']['beta']
        margin = config['loss']['margin']

        self.num_epochs = config['train']['num_epochs']
        num_workers = config['train']['num_workers']
        batch_size = config['train']['batch_size']
        learning_rate = config['train']['learning_rate']
        eps = float(config['train']['eps'])
        weight_decay = float(config['train']['weight_decay'])
        momentum = float(config['train']['momentum'])
        lr_step = config['train']['lr_step']
        lr_scale = config['train']['lr_scale']
        self.log_interval = config['train']['log_interval']
        self.logger = Logger(config['train']['log'])

        width = config['data']['width']
        height = config['data']['height']


        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.model = SigNet().to(self.device)
        self.criterion = ContrastiveLoss(alpha=alpha, beta=beta, margin=margin).to(self.device)
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=learning_rate, eps=eps, weight_decay=weight_decay,
                                  momentum=momentum)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, lr_step, lr_scale)

        image_transform = transforms.Compose([
            transforms.Resize((height, width)),
            ImageOps.invert,
            transforms.ToTensor(),
            # TODO: add normalize
        ])

        self.train_loader = get_data_loader(data_dir=config['data']['data_dir'], is_train=True, batch_size=batch_size,
                                      image_transform=image_transform, num_workers=num_workers)
        self.val_loader = get_data_loader(data_dir=config['data']['data_dir'], is_train=False, batch_size=batch_size,
                                     image_transform=image_transform, num_workers=num_workers)
        os.makedirs('checkpoints', exist_ok=True)


    def train(self):
        self.model.train()
        running_loss = 0
        number_samples = 0

        for batch_idx, (x1, x2, y) in enumerate(self.train_loader):
            x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            x1, x2 = self.model(x1, x2)
            loss = self.criterion(x1, x2, y)
            loss.backward()
            self.optimizer.step()

            number_samples += len(x1)
            running_loss += loss.item() * len(x1)
            if (batch_idx + 1) % self.log_interval == 0 or batch_idx == len(self.train_loader) - 1:
                infor = '{}/{}: Loss: {:.4f}'.format(batch_idx + 1, len(self.train_loader), running_loss / number_samples)
                print(infor)
                self.logger.log(infor)
                running_loss = 0
                number_samples = 0

    def val(self):

        running_loss = 0
        number_samples = 0

        distances = []
        with torch.no_grad():
            self.model.eval()
            for batch_idx, (x1, x2, y) in enumerate(self.val_loader):
                x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)

                x1, x2 = self.model(x1, x2)
                loss = self.criterion(x1, x2, y)
                distances.extend(zip(torch.pairwise_distance(x1, x2, 2).cpu().tolist(), y.cpu().tolist()))

                number_samples += len(x1)
                running_loss += loss.item() * len(x1)

                if (batch_idx + 1) % self.log_interval == 0 or batch_idx == len(self.val_loader) - 1:
                    infor = 'Valid: {}/{}: Loss: {:.4f}'.format(batch_idx + 1, len(self.val_loader), running_loss / number_samples)
                    print(infor)
                    self.logger.log(infor)

            distances, y = zip(*distances)
            distances, y = torch.tensor(distances), torch.tensor(y)
            max_accuracy = accuracy(distances, y)
            print(f'Max accuracy: {max_accuracy}')
            return running_loss / number_samples, max_accuracy

    def run(self):
        for epoch in range(self.num_epochs):
            print('Epoch {}/{}'.format(epoch, self.num_epochs))
            print('Training', '-' * 20)
            self.train()
            print('Evaluating', '-' * 20)
            loss, acc = self.val()
            self.scheduler.step()

            to_save = {
                'model': self.model.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'optim': self.optimizer.state_dict(),
            }

            print('Saving checkpoint..')
            torch.save(to_save, 'checkpoints/epoch_{}_loss_{:.3f}_acc_{:.3f}.pt'.format(epoch, loss, acc))
        self.logger.close()
        print('Done')


if __name__ == "__main__":
    config_path = "./config/config.yml"
    trainer = Trainer(config_path)
    trainer.run()