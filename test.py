import torch
import torchvision
import torchvision.utils
import torch.utils.data as utils
from torch.utils.data import DataLoader, Dataset
from utils import imshow, show_plot
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pandas as pd
from PIL import Image
import os
import torchvision.transforms as transforms

# load the dataset
training_dir = "siamese_net-master/siamese-net/tr"
training_csv = "siamese_net-master/siamese-net/tr.csv"
testing_csv = "siamese_net-master/siamese-net/tee.csv"
testing_dir = "siamese_net-master/siamese-net/te"
batch_size = 32
epochs = 20
# print(os.listdir())
# preprocessing and loading the dataset
class SiameseDataset:
    def __init__(self, training_csv=None, training_dir=None, transform=None):
        # used to prepare the labels and images path
        self.train_df = pd.read_csv(training_csv)
        self.train_df.columns = ["image1", "image2", "label"]
        self.train_dir = training_dir
        self.transform = transform

    def __getitem__(self, index):

        # getting the image path
        image1_path = os.path.join(self.train_dir, self.train_df.iat[index, 0])
        image2_path = os.path.join(self.train_dir, self.train_df.iat[index, 1])
        # print(f"{image1_path}, {image2_path}", end=", ")
        # Loading the image
        img0 = Image.open(image1_path)
        img1 = Image.open(image2_path)
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        # Apply image transformations
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return (
            img0,
            img1,
            torch.from_numpy(
                np.array([int(self.train_df.iat[index, 2])], dtype=np.float32)
            ),
        )

    def __len__(self):
        return len(self.train_df)
def main():
    #create a siamese network
    class SiameseNetwork(nn.Module):
        def __init__(self):
            super(SiameseNetwork, self).__init__()
            
            # Setting up the Sequential of CNN Layers
            self.cnn1 = nn.Sequential(
                
                nn.Conv2d(1, 96, kernel_size=11,stride=1),
                nn.BatchNorm2d(96),
                #nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, stride=2),
                
                nn.Conv2d(96, 256, kernel_size=5,stride=1,padding=2),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                #nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),
                nn.MaxPool2d(3, stride=2),
                nn.Dropout2d(p=0.3),

                nn.Conv2d(256,384 , kernel_size=3,stride=1,padding=1),
                nn.BatchNorm2d(384),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(384,256 , kernel_size=3,stride=1,padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, stride=2),
                nn.Dropout2d(p=0.3),

            )
            
            # Defining the fully connected layers
            self.fc1 = nn.Sequential(
                nn.Linear(30976, 1024),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.5),
                
                nn.Linear(1024, 128),
                nn.ReLU(inplace=True),
                
                nn.Linear(128,2))
            


        def forward_once(self, x):
            # Forward pass 
            output = self.cnn1(x)
            output = output.view(output.size()[0], -1)
            output = self.fc1(output)
            return output

        def forward(self, input1, input2):
            # forward pass of input 1
            output1 = self.forward_once(input1)
            # forward pass of input 2
            output2 = self.forward_once(input2)
            return output1, output2

    test_dataset = SiameseDataset(
        training_csv=testing_csv,
        training_dir=testing_dir,
        transform=transforms.Compose(
            [transforms.Resize((105, 105)), transforms.ToTensor()]
        ),
    )

    test_dataloader = DataLoader(test_dataset, num_workers=6, batch_size=1, shuffle=False)
    # net = SiameseNetwork().cuda()
    model_state_dict  = torch.load('siamese_net-master/siamese-net/content/model1.pth')
    net = SiameseNetwork()
    net.load_state_dict(model_state_dict)
    net.cuda()
    net.eval()

    count = 0

    for i, data in enumerate(test_dataloader, 0):
        x0, x1, label = data
        concat = torch.cat((x0, x1), 0)
        output1, output2 = net(x0.to('cuda'), x1.to('cuda'))

        eucledian_distance = F.pairwise_distance(output1, output2)
        
        if label == torch.FloatTensor([[0]]):
            label = "OriginalPair"
        else:
            label = "QuestionedPair"

        # imshow(torchvision.utils.make_grid(concat))
        # print("Predicted Eucledian Distance:-", eucledian_distance.item())
        print(i,label,",",eucledian_distance.item())
        # print("Actual Label:-", label)
        count = count + 1
        # print(count)
        if count == 108:
            break

if __name__ == '__main__':
    main()