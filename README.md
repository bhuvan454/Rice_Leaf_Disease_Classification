# Rice Leaf Disease Classification using PyTorch

![Rice Leaf Disease Classification](rice_leaf_image.jpg)

## About

This project aims to classify rice leaf diseases using a deep learning model built with PyTorch. The model is based on the ResNet architecture and has been trained to identify various diseases affecting rice plants based on leaf images.

## Dataset

The dataset used for training and evaluation is the [Rice Leaf Disease dataset](link-to-dataset), which contains a diverse set of labeled images for different rice leaf diseases. You can obtain the dataset from [source link].

## Requirements

- Python 3.x
- PyTorch
- torchvision
- NumPy
- Matplotlib
- [Additional dependencies, if any]

You can install the required packages using the provided `requirements.txt` file.

## Installation

1. Clone this repository:

   git clone https://github.com/bhuvan454/Rice_Leaf_Disease_Classification.git
   cd LeafDisease_classfication

2. 


## Rice Leaf Disease Classificaiton Using Pytorch

In this code repostitory, a classic resetnet based image classification task is prefromed with the pytorch. Whats new in this repository? I have utilized the GPU and CPU to parallelize the all operations. Pytorch is amazingly good at the parallleizing the model traing, and inferance, where as in the most of the pipelines the bottleneck could be the preprocessing step. So, to tackle that issue, I have used the python multiprocessing inbuilt funtions to create the multi worker pool to speed up those non-gpu tasks. 

So for this dataset I have used rice leaf disease dataset from the  <a href = "https://www.kaggle.com/datasets/vbookshelf/rice-leaf-diseases" target="_blank"> kaggle </a>, in this dataset, we have three classes of the diseased rice leaf. Leaf smut, Brown spot, Bacterial leaf blight are the classes. 

![image](data/figures/sample_images.png)

├── fedlab
│   ├── contrib
│   ├── core
│   ├── models
│   └── utils
├── datasets
│   └── ...
├── examples
│   ├── asynchronous-cross-process-mnist
│   ├── cross-process-mnist
│   ├── hierarchical-hybrid-mnist
│   ├── network-connection-checker
│   ├── scale-mnist
│   └── standalone-mnist
└── tutorials
    ├── communication_tutorial.ipynb
    ├── customize_tutorial.ipynb
    ├── pipeline_tutorial.ipynb
    └── ...