import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import utils
from data import Data
from model import Model
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', required=True)
parser.add_argument('--dataset_name', required=True, default='mnist')
parser.add_argument('--init_labeled_percent', required=False, default=0.01)
parser.add_argument('--device', required=False, default='cuda')
parser.add_argument('--batch_size', required=False, default=8)
parser.add_argument('--lr', required=False, default=0.001)
args = parser.parse_args()


def train_val(train_dl, validation_dl, epochs=5):
    """
    train model, return best validation accuracy
    """

    best_accuracy = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = []
        for imgs, labels in train_dl:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            preds = model(imgs)
            batch_loss = loss_fn(preds, labels)
            train_loss.append(batch_loss.item())

            batch_loss.backward()
            optimizer.step()
        train_loss = np.array(train_loss).mean()

        model.eval()
        with torch.no_grad():
            class_correct = 0
            val_loss = []
            for imgs, labels in validation_dl:
                imgs, labels = imgs.to(device), labels.to(device)

                preds = model(imgs)
                batch_loss = loss_fn(preds, labels)
                val_loss.append(batch_loss.item())

                y_preds = torch.max(torch.softmax(preds, dim=1), dim=1)[1]
                c = (labels == y_preds)
                class_correct += c.tolist().count(True)
            val_loss = np.array(val_loss).mean()
            accuracy = class_correct / (validation_dl.batch_size * len(validation_dl))
            best_accuracy.append(accuracy)

        print('epoch: {}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}'
              .format(epoch, train_loss, val_loss, accuracy))

    best_accuracy = max(best_accuracy)

    return best_accuracy


def test_unlabeled(unlabeled_ds, threshold=0.99):
    """
    if predicted label' confidence is higher then threshold,
    labeling y as predicted y
    """

    unlabeled = torch.stack([i[0] for i in unlabeled_ds])
    unlabeled_dl = DataLoader(unlabeled, batch_size=batch_size)

    pseudo_labeled = []
    index = []
    model.eval()
    with torch.no_grad():
        for i, imgs in enumerate(unlabeled_dl):
            imgs = imgs.to(device)

            preds = model(imgs)
            preds = torch.softmax(preds, dim=1)
            conf_preds, y_preds = torch.max(preds, dim=1)
            for j, confidence in enumerate(conf_preds):
                if confidence >= threshold:
                    pseudo_labeled.append((imgs[j].cpu(), y_preds[j].item()))
                    idx = (i * unlabeled_dl.batch_size) + j
                    index.append(idx)

    print(f'pseudo_labeled size: {len(pseudo_labeled)}')

    return pseudo_labeled, index


def update_dataset(pseudo_labeled, index):
    """
    add pseudo_labeled to labeled_dataset
    """

    for i in sorted(index, reverse=True):
        del unlabeled_ds[i]

    labeled_ds.extend(pseudo_labeled)

    print(f'labeled size: {len(labeled_ds)}, unlabeled size: {len(unlabeled_ds)}')

    return labeled_ds, unlabeled_ds


def evaluation():
    model.eval()
    with torch.no_grad():
        class_correct = 0
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)
            y_preds = torch.max(torch.softmax(preds, dim=1), dim=1)[1]
            c = (labels == y_preds)
            class_correct += c.tolist().count(True)
        accuracy = class_correct / (dataloader.batch_size * len(dataloader))
    print(f'evaluation accuracy: {accuracy:.4f}')
    return accuracy


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'cuda' else torch.device('cpu')
    batch_size = args.batch_size
    data_path = args.data_path
    dataset_name = args.dataset_name
    init_labeled_percent = float(args.init_labeled_percent)
    lr = args.lr

    # Fix random seed
    utils.fix_seed()

    # Load data, Separate data from labels
    data = Data(data_path, dataset_name)
    dataset, labeled_ds, unlabeled_ds, img_info = data.get_dataset(init_labeled_percent)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # settings for training model
    model = Model(img_info['channels'], img_info['size']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # 1-step pseudo labeling
    train_dl, validation_dl = utils.train_test_split(labeled_ds.copy())
    train_val(train_dl, validation_dl)
    evaluation()
    pseudo_labeled, index = test_unlabeled(unlabeled_ds)
    labeled_ds, unlabeled_ds = update_dataset(pseudo_labeled, index)

    # show pseudo label's confidence
    utils.find_wrong_pseudo_labeled(dataset.copy(), labeled_ds.copy(), data.labeled_size)
