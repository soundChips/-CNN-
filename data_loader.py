import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(data_dir, batch_size=32, input_size=(224, 224)):
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    datasets_dict = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform)
                     for x in ['train', 'val', 'test']}
    dataloaders_dict = {x: DataLoader(datasets_dict[x], batch_size=batch_size, shuffle=(x == 'train'))
                        for x in datasets_dict.keys()}

    return dataloaders_dict, datasets_dict['train'].classes
