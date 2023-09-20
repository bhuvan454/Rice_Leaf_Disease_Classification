import os
from torch.utils.data import DataLoader 


def CustomDataLoader(dataset, batch_size, shuffle, num_workers):
    '''
    Custom dataloader function for loading the dataset. 

    Args:
        dataset (torch.utils.data.Dataset): dataset to be loaded
        batch_size (int): batch size for training
        shuffle (bool): whether to shuffle the dataset
        num_workers (int): number of workers for loading the dataset

    Returns:
        dataloader (torch.utils.data.DataLoader): dataloader for the dataset

    '''
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader
