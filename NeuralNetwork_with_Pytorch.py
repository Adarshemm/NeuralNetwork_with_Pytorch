import torch 
from PIL import Image

#provides a set of high-level abstractions for building and training neural networks. 
#Activation functions like ReLU, Sigmoid, and Tanh are available, 
#Common loss functions like CrossEntropyLoss, MSELoss, etc., are also part of nn
from torch import nn, save, load

# In the context of deep learning, optimizers are algorithms or 
#methods used to update the parameters (weights and biases) of a neural network during training
from torch.optim import Adam

# it  is a utility class used to load and iterate over 
#batches of data during the training or evaluation of a neural network.
from torch.utils.data import DataLoader

# is a package that provides datasets, models, and transformations specific to computer vision tasks.
from torchvision import datasets

#is used to convert a PIL Image or a NumPy array representing an image to a PyTorch tensor.
from torchvision.transforms import ToTensor

# Get data
train = datasets.MNIST(root='data', download=True, train=True, transform=ToTensor())
dataset = DataLoader(train, 32)
#images from MNIST are in the shape (1, 28, 28) and classes are 0-9 i.e predicting the image 0,1,2,3...9 using cnn and back prop

#image classifier at neural network
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3,3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3)),
            nn.ReLU(),
            nn.Flatten(),               
            nn.Linear(64*(28-6)*(28-6), 10)                   #so each neural network is going to shade off 2 pixels each so 3 nn shades off 6 pixels
        )

    def forward(self, x):
        return self.model(x)

#instance of neural network, loss and optmiser 
clf = ImageClassifier().to('cpu')
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# # Training flow
# if __name__ == "__main__":
#     for epoch in range(10): #train for 10 epochs
#         for batch in dataset:
#             X, y = batch
#             X, y = X.to('cpu'), y.to('cpu')
#             yhat = clf(X)
#             loss = loss_fn(yhat, y)

#             # Apply back Propagation
#             opt.zero_grad()
#             loss.backward()
#             opt.step()

#         print(f"Epoch:{epoch} loss is {loss.item()}")

#     with open('model_state.pt', 'wb') as f:
#         save(clf.state_dict(), f)

with open('model_state.pt', 'rb') as f: 
    clf.load_state_dict(load(f))  

    img = Image.open('img_2.jpg') 
    img_tensor = ToTensor()(img).unsqueeze(0).to('cpu')

    print(torch.argmax(clf(img_tensor)))