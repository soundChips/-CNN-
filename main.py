import torch
from data_loader import get_data_loaders
from model import get_model
from train import train_model
from evaluate import evaluate_model

def main():
    data_dir = 'data'
    batch_size = 16
    num_epochs = 10
    input_size = (224, 224)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataloaders, class_names = get_data_loaders(data_dir, batch_size, input_size)
    model = get_model(num_classes=len(class_names))

    print("Training Model...")
    model = train_model(model, dataloaders, num_epochs, device)

    print("Evaluating Model...")
    evaluate_model(model, dataloaders['test'], device, class_names)

if __name__ == '__main__':
    main()
