import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader


# 1. Define the CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # Conv layer 1 -> ReLU -> MaxPool
        x = self.pool(self.relu(self.conv2(x)))  # Conv layer 2 -> ReLU -> MaxPool
        x = x.view(-1, 64 * 7 * 7)  # Flatten the tensor for the fully connected layers
        x = self.relu(self.fc1(x))  # Fully connected layer 1 -> ReLU
        x = self.fc2(x)  # Fully connected layer 2
        return x


# 2. Load the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize using the dataset's mean and std
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# 3. Initialize the network, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)

criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Train the CNN
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Accumulate statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, '
          f'Accuracy: {100 * correct / total:.2f}%')

# 5. Evaluate the CNN on the test set
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

print(f'Test Accuracy: {100 * correct / total:.2f}%')


# Function to test the model on a specific image
def test_model_on_index(index):
    model.eval()
    with torch.no_grad():
        # Get the image and label from the test dataset
        image, label = test_dataset[index]

        # Add a batch dimension and move to device
        image = image.unsqueeze(0).to(device)

        # Make the prediction
        output = model(image)
        _, predicted = output.max(1)

        # Display the image and the prediction
        plt.imshow(image.cpu().squeeze(), cmap='gray')
        plt.title(f'Predicted: {predicted.item()}, Actual: {label}')
        plt.axis('off')
        plt.show()


# Example of testing the model on a specific index
# Replace the index with any number from 0 to 9999 (total number of test images)
while True:
    index_to_test = int(input("Enter the index of the image to test (0-9999): "))
    test_model_on_index(index_to_test)
