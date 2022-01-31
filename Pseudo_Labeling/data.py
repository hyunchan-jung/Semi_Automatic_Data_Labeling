import torchvision
from torchvision import transforms


class Data:
    def __init__(self, data_path, dataset_name):
        self.labeled_size = None
        self.img_info = {'channels': None, 'size': None}
        self.data_path = data_path
        self.dataset_name = dataset_name

    def get_dataset(self, labeled_percent):
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        if self.dataset_name.lower() == 'mnist':
            self.img_info['channels'] = 1
            self.img_info['size'] = 28
            dataset = list(torchvision.datasets.MNIST(self.data_path,
                                                      download=True,
                                                      train=True,
                                                      transform=transform
                                                      ))
        elif self.dataset_name.lower() == 'cifar10':
            self.img_info['channels'] = 3
            self.img_info['size'] = 32
            dataset = list(torchvision.datasets.CIFAR10(self.data_path,
                                                        download=True,
                                                        train=True,
                                                        transform=transform
                                                        ))
        elif self.dataset_name.lower() == 'stl10':
            self.img_info['channels'] = 3
            self.img_info['size'] = 96
            dataset = list(torchvision.datasets.STL10(self.data_path,
                                                      download=True,
                                                      split='train',
                                                      transform=transform
                                                      ))
        else:
            print('select dataset in MNIST,STL10,CIFAR10')
            exit()

        self.labeled_size = int(len(dataset) * labeled_percent)
        print(f'labeled {labeled_percent*100} % data:', self.labeled_size)

        labeled_ds = dataset[:self.labeled_size]
        unlabeled_ds = dataset[self.labeled_size:]
        print(f'labeled_data: {len(labeled_ds)}, unlabeled_data: {len(unlabeled_ds)}')

        return dataset, labeled_ds, unlabeled_ds, self.img_info
