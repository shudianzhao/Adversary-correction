from model import ResNet8_3
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


def main():
    # Load the MNIST dataset
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform = transforms.Compose([transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(*stats, inplace=True)])    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    # Instantiate the model
    model = ResNet8_3()

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Training loop
    for epoch in range(10):  # You can adjust the number of epochs
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:  # Print every 100 batches
                print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100:.3f}')
                running_loss = 0.0

    print('Finished Training')


    # Evaluation on test set
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy on the test set: {100 * correct / total:.2f}%')


    # save the trained model
    torch.save(model, './NN_models/cifar10_ResNet8_normDataAug_noConv3_Epoch10.pth')


    # Create an empty dictionary to store the model parameters
    model_params = {}

    # Save the weights and biases for each layer in the model
    for name, param in model.named_parameters():
        if param.requires_grad:
            model_params[name] = param.detach().numpy()


    # Save the dictionary containing model parameters as a NumPy .npz file
    np.savez('./NN_models/cifar10_ResNet8_normDataAug_noConv3_Epoch10.npz', **model_params)


if __name__ == '__main__':
    main()
