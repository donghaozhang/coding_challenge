import os.path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

cifar10 = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def invTrans(img):
    '''
    Perform inverse transformation of image.
    '''
    return T.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
    )(img)


class Runner:
    def __init__(
            self,
            model: nn.Module = None,
            optimizer: Optimizer = None,
            train_loader: DataLoader = None,
            val_loader: DataLoader = None,
            test_loader: DataLoader = None,
            scheduler=None,
            device: torch.device = torch.device('cpu'),
            input_path='cifar10/test_tiny'
    ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.seed = 1024
        self.save_dir = './results'
        self.current_epoch = 0
        self.input_path = input_path
        self.build_model()

    def build_model(self, checkpoint=None):
        if checkpoint is not None:
            self.load_model(checkpoint)
        self.model.to(self.device)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc='Training')
        for i, data in enumerate(pbar):
            images, labels = [d.to(self.device) for d in data]
            logits = self.model(images)
            loss = self.model.compute_loss(logits, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            loss_avg = total_loss / (i + 1)
            pbar.set_postfix({
                'loss': loss_avg,
                'lr': self.optimizer.param_groups[0]['lr']
            })
            if self.scheduler is not None:
                self.scheduler.step()
        return total_loss / len(self.train_loader)

    def train_model(self, epochs=10, start_epoch=0):
        for epoch in range(start_epoch, epochs):
            self.model.train()
            self.current_epoch = epoch
            print(f'Epoch: {epoch}:')
            train_loss = self.train_epoch()
            self.save_model(os.path.join(self.save_dir, "checkpoints", f'epoch_{epoch}.pth'))
            val_loss = self.val_epoch()
            print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print('Training finished. ')

    def val_epoch(self):
        self.model.eval()
        final_loss = 0
        pbar = tqdm(self.val_loader, desc='Validation')
        with torch.no_grad():
            for i, data in enumerate(pbar):
                images, labels = [d.to(self.device) for d in data]
                logits = self.model(images)
                loss = self.model.compute_loss(logits, labels)
                final_loss += loss.item()
                loss_avg = final_loss / (i + 1)
                pbar.set_postfix({
                    'loss': loss_avg,
                })
        return final_loss / len(self.val_loader)

    def test_model(self):
        self.model.eval()
        y_pred = []
        y_true = []
        y_pred_probs = []
        y_labels_oneh = []
        error_count = 0
        pbar = tqdm(self.test_loader, desc='Testing')
        with torch.no_grad():
            for i, data in enumerate(pbar):
                images, labels = [d.to(self.device) for d in data]
                outputs, pred_prob = self.predict(images)
                y_pred.extend(outputs.data.cpu().numpy())
                y_true.extend(labels.data.cpu().numpy())
                y_pred_prob = pred_prob.cpu().detach().numpy()
                y_pred_probs.extend(y_pred_prob)
                # Convert labels to one hot encoding
                y_label_oneh = torch.nn.functional.one_hot(labels, num_classes=10)
                y_label_oneh = y_label_oneh.cpu().detach().numpy()
                y_labels_oneh.extend(y_label_oneh)

        y_pred_probs = np.array(y_pred_probs).flatten()
        y_labels_oneh = np.array(y_labels_oneh).flatten()
        return np.array(y_pred), np.array(y_true), np.array(y_pred_probs), np.array(y_labels_oneh)

    def save_false_positives_v2(self, save_wrong_images=True, save_feature=True):
        self.model.eval()
        y_pred = []
        y_true = []
        error_count = 0
        pbar = tqdm(self.test_loader, desc='Testing')
        with torch.no_grad():
            for i, data in enumerate(pbar):
                images, labels = [d.to(self.device) for d in data]
                outputs, _ = self.predict(images)
                y_pred.extend(outputs.data.cpu().numpy())
                y_true.extend(labels.data.cpu().numpy())

                if save_wrong_images:
                    # save wrong images
                    for j, (output, label) in enumerate(zip(outputs, labels)):
                        error_count += 1
                        if output != label:
                            image = invTrans(images[j].data).cpu().numpy().transpose(1, 2, 0) * 255
                            image = Image.fromarray(image.astype(np.uint8))
                            image.save(os.path.join("results/false_positives", cifar10[output],
                                                    f'{cifar10[label]}_{error_count}.png'))

                if save_feature:
                    for j, (output, label) in enumerate(zip(outputs, labels)):
                        if output != label:
                            feature = self.model.get_feature(images[j].unsqueeze(0))
                            feature = feature.data.cpu().numpy()
                            np.save(
                                os.path.join("results/false_positives_feature", cifar10[output],
                                             f'{cifar10[label]}_{error_count}.npy'),
                                feature
                            )

        return np.array(y_pred), np.array(y_true)

    def predict(self, x):
        self.model.eval()
        x = x.to(self.device)
        with torch.no_grad():
            pred = self.model(x)
            y_pred = (torch.max(torch.exp(pred), 1)[1])
            sm = nn.Softmax(dim=1)
            pred_prob = sm(pred)

        return y_pred, pred_prob

    def save_false_positives(self):
        class_to_label = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6,
                          'horse': 7, 'ship': 8, 'truck': 9}
        label_class = {v: k for k, v in class_to_label.items()}
        # load the cifar10
        # Normalize the images by the imagenet mean/std since the nets are pretrained
        transform = T.Compose([
            T.Resize(64),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        for class_name in os.listdir(self.input_path):
            class_folder = os.path.join(self.input_path, class_name)
            # print(class_folder)
            if not os.path.isdir(class_folder):
                continue
            pbar = tqdm(os.listdir(class_folder), desc=class_name)
            # Loop over the images in the class folder
            for i, file_name in enumerate(pbar):
                image_path = os.path.join(class_folder, file_name)
                image_ori = Image.open(image_path)
                image = transform(image_ori)
                image = image.to(self.device)
                image = image.unsqueeze(0)
                output, _ = self.predict(image)
                output = output.cpu().detach().numpy()[0]
                true_label = class_to_label[class_name]
                if true_label != output:
                    # print('true_label', class_name, 'prediction', label_class[output], image_path)
                    path = os.path.join('results/false_positives', cifar10[output],
                                        f'{class_name}_{os.path.splitext(file_name)[0]}.png')
                    image_ori.save(path)
        return None

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        try:
            self.model.load_state_dict(checkpoint['model'])
        except RuntimeError:
            print('Loaded model is not compatible with current model.')
            return

        try:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint['scheduler'])
        except KeyError:
            print('Optimizer and scheduler not loaded.')

    def save_model(self, path):
        state_dict = {
            'model': self.model.state_dict(),
        }
        if self.optimizer is not None:
            state_dict['optimizer'] = self.optimizer.state_dict()
        if self.scheduler is not None:
            state_dict['scheduler'] = self.scheduler.state_dict()
        torch.save(state_dict, path)
        print(f'Model saved to {path}.')
