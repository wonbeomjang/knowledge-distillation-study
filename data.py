from torch.utils.data import DataLoader
from torchvision import transforms


def get_loader(config):
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(config.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset_train = config.dataset(config.data_dir, train=True, transform=train_transform, download=True)
    dataset_test = config.dataset(config.data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.num_workers)
    test_loader = DataLoader(dataset_test, batch_size=config.batch_size, shuffle=False,
                             num_workers=config.num_workers)

    return train_loader, test_loader
