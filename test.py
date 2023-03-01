import argparse
import itertools

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from matplotlib import pyplot as plt

from models.base_model import BaseModel
from runner import Runner
from utils import fix_seed, make_project_dirs, cifar10
import matplotlib.patches as mpatches


def parse_args():
    parser = argparse.ArgumentParser(description='Main program for ECE Script.')
    parser.add_argument('--model', type=str, default='tiny', help="model to use.",
                        choices=['tiny', 'small', 'medium', 'large'])
    parser.add_argument('--checkpoint', type=str, default="./results/checkpoints/epoch_3.pth",
                        help="checkpoint to resume.")
    parser.add_argument('--train_epochs', type=int, default=3, help="number of epochs to train.")
    parser.add_argument('--test_dir', type=str, default='cifar10/test', help="path to test data.")
    parser.add_argument('--run_train', action='store_true', default=False, help="only test the model.")
    opt = parser.parse_args()
    return opt


def build_runner(opt):
    model = BaseModel(model_type=opt.model)
    transform = T.Compose([
        T.Resize(64),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if opt.run_train:
        data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        train_data, val_data = torch.utils.data.random_split(data, [45000, 5000])
        # train_data = torchvision.datasets.ImageFolder(root='cifar10/train', transform=transform)
        # val_data = torchvision.datasets.ImageFolder(root='cifar10/val', transform=transform)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=8)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=False, num_workers=8)
    else:
        train_loader = None
        val_loader = None

    # test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_data = torchvision.datasets.ImageFolder(root=opt.test_dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=4, shuffle=False, num_workers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.0001)

    runner = Runner(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=torch.device('cuda'),
        input_path=opt.test_dir
    )
    return runner


def compute_confusion_matrix(pred, gt, num_classes):
    confusion_matrix = np.zeros((num_classes, num_classes))
    for i in range(len(pred)):
        confusion_matrix[gt[i], pred[i]] += 1
    return confusion_matrix


def plot_confusion_matrix(confusion_matrix, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

    plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=30)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else '0.1f'
    thresh = confusion_matrix.max() / 2.
    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        plt.text(j, i, format(confusion_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if confusion_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # increase the size of the plot
    fig = plt.gcf()
    fig.set_size_inches(8, 7)

    plt.savefig('results/confusion_matrix.png')
    plt.show()


def main(opt):
    runner = build_runner(opt)
    if opt.checkpoint is not None:
        runner.load_model(opt.checkpoint)
    if opt.run_train:
        runner.train_model(epochs=opt.train_epochs)
    print('---Perform inference on a folder of example images---')
    pred, gt, pred_probs, labels_oneh = runner.test_model()
    print('---Compute and plot the confusion matrix---')
    cm2 = compute_confusion_matrix(pred, gt, 10)
    plot_confusion_matrix(cm2, classes=cifar10, normalize=False)


if __name__ == '__main__':
    fix_seed(1024)
    opt = parse_args()
    make_project_dirs('results')
    main(opt)