
import torch

def train_per_epoch(model,device, train_loader, criterion, optimizer, epoch,log_interval):
    model.train()
    losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        #print(data.shape)
        optimizer.zero_grad()
        output = model(data)
        #print(output.shape)
        loss = criterion(output, target)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return losses


def train_model(model,device, train_loader, criterion, optimizer, epochs, log_interval,save_model_path):
    train_losses = []
    for epoch in range(1, epochs + 1):
        train_loss = train_per_epoch(model,device, train_loader, criterion, optimizer, epoch,log_interval)
        train_losses.extend(train_loss)

        if epoch % 10 == 0 and save_model_path is not None:
            torch.save(model.state_dict(), save_model_path)
    return train_losses