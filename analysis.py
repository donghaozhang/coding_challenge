import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans

from models.base_model import BaseModel
from utils import cifar10
from sklearn.manifold import TSNE


def read_all_data(dir_path):
    """
    Read all data from a dir
    :param dir_path: directory path
    :return: list of feature data
    """
    data = [[] for _ in range(10)]
    true_labels = []
    for file in os.listdir(dir_path):
        if not file.endswith(".npy"):
            continue
        file_path = os.path.join(dir_path, file)
        cls = file.split("_")[0]
        cls_id = cifar10.index(cls)
        print(cls_id)
        feature = np.load(file_path)[0]
        new_feature = np.zeros((feature.shape[0] + 1))
        new_feature[:feature.shape[0]] = feature
        new_feature[-1] = cls_id
        # data.append(feature)
        data[cls_id].append(feature)
    return data


def main():
    true_class = "car"
    true_class_id = cifar10.index(true_class)
    data_dir = f"./results/false_positives_feature/{true_class}"
    data = read_all_data(data_dir)
    total = sum([len(d) for d in data])
    print(f"True class: {true_class}, total: {total}")
    print('Error ratio for each error class:')
    for i, d in enumerate(data):
        if i == true_class_id:
            continue
        ratio = len(d) / total * 100
        print(f"class {cifar10[i]}: {ratio:.2f}%")

    # plot error ratio
    plt.figure()
    plt.title(f"Error ratio for {true_class}")
    plt.xlabel("Error class")
    plt.ylabel("Error ratio (%)")
    error_classes = [c for c in cifar10 if c != true_class]
    error_ratio = [len(d) / total * 100 for i, d in enumerate(data) if i != true_class_id]
    plt.bar(error_classes, error_ratio)
    plt.savefig(f"./results/error_ratio_{true_class}.png")
    plt.show()
    # by looking at the error ratio, we can decide k for k-means
    all_features = []
    all_labels = []
    for x in data:
        for y in x:
            all_features.append(y)
            all_labels.append(x)
    kmeans = KMeans(n_clusters=3, random_state=0, max_iter=500).fit(all_features)
    features = [torch.Tensor(d) for d in kmeans.cluster_centers_]

    # model = BaseModel()
    # state = torch.load("./results/checkpoints/epoch_3.pth", map_location="cpu")['model']
    # model.load_state_dict(state)
    # model.eval()
    # for i, feature in enumerate(features):
    #     x = model.classify(feature.unsqueeze(0))
    #     y_pred = (torch.max(torch.exp(x), 1)[1])
    #     print(f"Cluster {i} center: {cifar10[y_pred.item()]}")


if __name__ == '__main__':
    main()