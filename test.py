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


def compute_exact_calibration_error(pred, gt, num_classes):
    """
    Compute the exact calibration error.
    :param pred: predicted logits
    :param gt: ground truth labels
    :param num_classes: number of classes
    :return: exact calibration error
    """
    confusion_matrix = compute_confusion_matrix(pred, gt, num_classes)
    row_sum = np.sum(confusion_matrix, axis=1)
    col_sum = np.sum(confusion_matrix, axis=0)
    accuracy = np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)
    ece = 0
    for i in range(num_classes):
        ece += np.abs(row_sum[i] / np.sum(row_sum) - col_sum[i] / np.sum(col_sum))
    ece = ece / 2
    return ece, accuracy


def calc_bins(pred_probs, labels_oneh):
    # Assign each prediction to a bin
    num_bins = 10
    bins = np.linspace(0.1, 1, num_bins)
    binned = np.digitize(pred_probs, bins)

    # Save the accuracy, confidence and size of each bin
    bin_accs = np.zeros(num_bins)
    bin_confs = np.zeros(num_bins)
    bin_sizes = np.zeros(num_bins)

    for bin in range(num_bins):
        bin_sizes[bin] = len(pred_probs[binned == bin])
    if bin_sizes[bin] > 0:
      bin_accs[bin] = (labels_oneh[binned==bin]).sum() / bin_sizes[bin]
      bin_confs[bin] = (pred_probs[binned==bin]).sum() / bin_sizes[bin]

    return bins, binned, bin_accs, bin_confs, bin_sizes


def get_metrics(pred_probs, labels_oneh):
    ECE = 0
    MCE = 0
    bins, _, bin_accs, bin_confs, bin_sizes = calc_bins(pred_probs, labels_oneh)

    for i in range(len(bins)):
        abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
        ECE += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif
        MCE = max(MCE, abs_conf_dif)

    return ECE, MCE


def draw_reliability_graph(pred_probs, labels_oneh):
    ECE, MCE = get_metrics(pred_probs, labels_oneh)
    bins, _, bin_accs, _, _ = calc_bins(pred_probs, labels_oneh)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()

    # x/y limits
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1)

    # x/y labels
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')

    # Create grid
    ax.set_axisbelow(True)
    ax.grid(color='gray', linestyle='dashed')

    # Error bars
    plt.bar(bins, bins, width=0.1, alpha=0.3, edgecolor='black', color='r', hatch='\\')

    # Draw bars and identity line
    plt.bar(bins, bin_accs, width=0.1, alpha=1, edgecolor='black', color='b')
    plt.plot([0, 1], [0, 1], '--', color='gray', linewidth=2)

    # Equally spaced axes
    plt.gca().set_aspect('equal', adjustable='box')

    # ECE and MCE legend
    ECE_patch = mpatches.Patch(color='green', label='ECE = {:.2f}%'.format(ECE * 100))
    MCE_patch = mpatches.Patch(color='red', label='MCE = {:.2f}%'.format(MCE * 100))
    plt.legend(handles=[ECE_patch, MCE_patch])

    plt.savefig('results/calibrated_graph.png', bbox_inches='tight')
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
    print('---Compute the Expected Calibration Error (ECE) and Max Calibration Error (MCE)---')
    ECE, MCE = get_metrics(pred_probs, labels_oneh)
    print('Exact calibration error: {:.2f}%'.format(ECE * 100), 'Max Calibration Error: {:.2f}%'.format(MCE * 100))
    draw_reliability_graph(pred_probs, labels_oneh)


if __name__ == '__main__':
    fix_seed(1024)
    opt = parse_args()
    make_project_dirs('results')
    main(opt)