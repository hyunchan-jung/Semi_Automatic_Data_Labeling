import random
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm


def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_test_split(labeled_ds):
    """
    split train_ds and validation_ds with random index of labeled_ds
    """

    size = len(labeled_ds)
    validation_idx = sorted(list(np.random.choice(range(size), int(size * 0.2), replace=False)),
                            reverse=True)
    validation_ds = [labeled_ds[i] for i in validation_idx]
    for i in validation_idx:
        del labeled_ds[i]
    train_ds = labeled_ds

    print(f'train data size: {len(train_ds)}, validation data size: {len(validation_ds)}')

    train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)
    validation_dl = DataLoader(validation_ds, batch_size=8, shuffle=False)

    return train_dl, validation_dl


def find_wrong_pseudo_labeled(mnist_ds, labeled_ds, labeled_size):
    mnist_ds.sort(key=lambda x: x[0].sum())
    labeled_ds.sort(key=lambda x: x[0].sum())

    idx = 0
    wrong_label_cnt = 0
    for img, label in tqdm(labeled_ds[labeled_size:]):
        for n, (img2, label2) in enumerate(mnist_ds[idx:]):
            if (img == img2).view(-1).tolist().count(False) == 0:
                idx += n
                if label != label2:
                    wrong_label_cnt += 1
                break
    print('pseudo labeled size: {}, wrong label count: {}, {:.2f}%'
          .format(len(labeled_ds)-labeled_size, wrong_label_cnt, wrong_label_cnt/(len(labeled_ds)-labeled_size)*100))