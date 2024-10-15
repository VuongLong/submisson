from domainbed.dataloaders.MNIST_Dataloader import MNIST_Test_Dataloader, MNISTDataloader
from domainbed.dataloaders.Standard_Dataloader import StandardDataloader, StandardValDataloader


train_dataloaders_map = {
    "PACS": StandardDataloader,
    "DomainNet": StandardDataloader,
    "MNIST": MNISTDataloader,
    "OfficeHome": StandardDataloader,
    "VLCS": StandardDataloader,
    'TerraIncognita': StandardDataloader
}

test_dataloaders_map = {
    "PACS": StandardValDataloader,
    "DomainNet": StandardValDataloader,
    "MNIST": MNIST_Test_Dataloader,
    "OfficeHome": StandardValDataloader,
    "VLCS": StandardValDataloader,
    'TerraIncognita': StandardValDataloader
}


def get_train_dataloader(name):
    if name not in train_dataloaders_map:
        raise ValueError("Name of train dataloader unknown %s" % name)

    def get_dataloader_fn(**kwargs):
        return train_dataloaders_map[name](**kwargs)

    return get_dataloader_fn


def get_test_dataloader(name):
    if name not in test_dataloaders_map:
        raise ValueError("Name of test dataloader unknown %s" % name)

    def get_dataloader_fn(**kwargs):
        return test_dataloaders_map[name](**kwargs)

    return get_dataloader_fn

def get_fourier_dataset(args, path, image_size=224, crop=False, jitter=0, from_domain='all', alpha=1.0, config=None):
    assert isinstance(path, list)
    names = []
    labels = []
    for p in path:
        name, label = dataset_info(p)
        names.append(name)
        labels.append(label)

    if config:
        image_size = config["image_size"]
        crop = config["use_crop"]
        jitter = config["jitter"]
        from_domain = config["from_domain"]
        alpha = config["alpha"]

    img_transform = get_pre_transform(image_size, crop, jitter)
    return FourierDGDataset(args, names, labels, img_transform, from_domain, alpha)