from model import ConvNetLight, ConvNet
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


def train(model, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target) # l1_regularization
        loss.backward(retain_graph=True)
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))



def main():
    train_kwargs = {'batch_size': 64}
    test_kwargs = {'batch_size': 1000}
    # transform = transforms.ToTensor()
    transform = transforms.Compose([transforms.ToTensor()])

    dataset1 = datasets.MNIST(root='./data', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST(root='./data', train=False,
                              transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # Create an instance of the CNN model
    model = ConvNetLight() # model = ConvNet()
    

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    t0 = time.time()
    for epoch in range(10):
        train(model, train_loader, optimizer, epoch)
        
        test(model, test_loader)       
        scheduler.step()

    print('Total training time (wall time) is: {}'.format(time.time() - t0))


    torch.save(model, './NN_models/mnist_ConvNetLight_norm01.pth')

    # Create an empty dictionary to store the model parameters
    model_params = {}

    # Save the weights and biases for each layer in the model
    for name, param in model.named_parameters():
        if param.requires_grad:
            model_params[name] = param.detach().numpy()

    # Save the dictionary containing model parameters as a NumPy .npz file
    np.savez('./NN_models/mnist_ConvNetLight_norm01.npz',
             **model_params)


if __name__ == '__main__':
    main()
