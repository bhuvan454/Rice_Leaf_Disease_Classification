import torch

def eval_per_epoch(model, device, test_loader, criterion, epoch, log_interval):
    model.eval()
    
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):

            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            if batch_idx % log_interval == 0:
                print('Eval Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(test_loader.dataset),
                    100. * batch_idx / len(test_loader), test_loss))
                
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)

    print('\nEval set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(

        test_loss, correct, len(test_loader.dataset),
        test_accuracy))
    
    return test_loss, test_accuracy


def eval_model(model, device, test_loader, criterion, log_interval):
    test_losses = []
    test_accuracies = []
    for epoch in range(1, 2):
        test_loss, test_accuracy = eval_per_epoch(model, device, test_loader, criterion, epoch, log_interval)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
    return test_losses, test_accuracies

