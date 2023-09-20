# Rice Leaf Disease Classificaiton Using Pytorch

This project aims to classify rice leaf diseases using a classic resetnet architecture based image classification task is prefromed with the pytorch. Whats new in this project? I have utilized the GPU and CPU to parallelize the all operations. Pytorch is amazingly good at the parallleizing the model traing, and inferance, where as in the most of the pipelines the bottleneck could be the preprocessing step. So, to tackle that issue, I have used the python multiprocessing inbuilt funtions to create the multi worker pool to speed up those non-gpu tasks. 

### About Dataset
So for this dataset I have used rice leaf disease dataset from the  <a href = "https://www.kaggle.com/datasets/vbookshelf/rice-leaf-diseases" target="_blank"> kaggle </a>, in this dataset, we have three classes of the diseased rice leaf. Leaf smut, Brown spot, Bacterial leaf blight are the classes. 

![image](data/figures/sample_images.png)


### Requirements

- Python 
- PIL
- PyTorch
- torchvision
- NumPy
- Matplotlib

You can install the required packages using the provided `requirements.txt` file. Soon will be updated!!!


### file structure

<!-- '''bash 

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

''' -->

### Installation

1. Clone this repository:

   - git clone https://github.com/bhuvan454/Rice_Leaf_Disease_Classification.git
   - cd LeafDisease_classfication

2.  pip install -r requirements.txt

3.  Run the training

    args.add_argument('--data_dir',type = str, help='path to dataset')
    args.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
    args.add_argument('--epochs', type=int, default=5, help='number of epochs to train (default: 100)')
    args.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')

    args.add_argument('--model', type=str, default='resnet18', help='model name, (default: resnet18) in [resnet18, resnet34, resnet50]')
    args.add_argument('--pretrained', type=bool, default=True, help='use pretrained model (default: True)')

    args.add_argument('--log_interval', type=int, default=10, help='how many batches to wait before logging training status')
    args.add_argument('--save_interval', type=int, default=10, help='how many epochs to wait before saving model weights')

    args.add_argument('--save_dir', type=str, default='../models', help='path to save weights')
    args.add_argument('--log_dir', type=str, default='../logs', help='path to save logs')

    args.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    args.add_argument('--gpu', type=int, default=0, help='GPU number to use (default: 0)')

```
$ cd ./src/
$ python main.py --data_dir '../data/raw' --batch_size 64 --epochs 10 --lr 0.001 --model resent18 --pretrained True --log_interval 10 --save_interval 10 --save_dir ../models/ --log_dir ../logs/ --seed 2109 --gpu 0

```