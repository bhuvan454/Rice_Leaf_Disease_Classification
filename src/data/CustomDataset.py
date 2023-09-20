
import os
from PIL import Image

class CustomDataset():
    '''
    Custom dataset class for loading images and labels the images, and also for getting the number of classes in the dataset. 

    Args:
        data_dir (str): path to the dataset directory
        transform (torchvision.transforms): transforms to be applied to the images

    Attributes:
        data_dir (str): path to the dataset directory
        class_names (list): list of class names
        image_paths (list): list of image paths
        image_labels (list): list of image labels

    Methods:
        __len__(): returns the length of the dataset
        __getitem__(idx): returns the image path and label of the image at index idx
        get_num_classes(): returns the number of classes in the dataset
    
    '''
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.class_names = os.listdir(data_dir)

        self.image_paths  = []
        self.image_labels = []

        for i, class_name in enumerate(self.class_names):
            for file_name in os.listdir(os.path.join(data_dir, class_name)):
                self.image_paths.append(os.path.join(data_dir,class_name,file_name))
                self.image_labels.append(i)

    def __len__(self):
        return len(self.image_paths)
    

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_label = self.image_labels[idx]
        image = Image.open(image_path)

        if self.transform is not None:
            image = self.transform(image)

        return image, image_label
    
    def get_num_classes(self):
        return len(self.class_names)
    