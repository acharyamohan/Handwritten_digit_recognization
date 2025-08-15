import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
import time
import tkinter as tk
from PIL import Image, ImageDraw

# Basic setup and device configuration
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DataLoader with optimized settings
def get_dataloader(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                     num_workers=2, pin_memory=True)

# Enhanced CNN Model with BatchNorm and Dropout
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(9216, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.dropout(self.fc1(x)))
        return self.fc2(x)

# GUI for Digit Drawing
class DigitRecognizer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MNIST Digit Recognizer")
        self.canvas = tk.Canvas(self, width=280, height=280, bg='white')
        self.canvas.pack()
        
        # Control buttons
        self.button_clear = tk.Button(self, text="Clear", command=self.clear_canvas)
        self.button_clear.pack()
        self.button_predict = tk.Button(self, text="Recognize", command=self.predict_digit)
        self.button_predict.pack()
        
        # Drawing setup
        self.canvas.bind('<B1-Motion>', self.paint)
        self.image = Image.new('L', (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)
        
        # Result display
        self.result = tk.Label(self, text="", font=("Helvetica", 48))
        self.result.pack()

    def paint(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x, y, x+10, y+10, fill='black', width=10)
        self.draw.ellipse([x, y, x+10, y+10], fill='black')

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new('L', (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.result.config(text="")

    def preprocess_image(self):
        image = self.image.resize((28, 28)).convert('L')
        image = np.array(image)
        image = 255 - image  # Invert colors
        image = image / 255.0
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        return image

    def predict_digit(self):
        processed_image = self.preprocess_image()
        with torch.no_grad():
            prediction = model(processed_image)
            digit = torch.argmax(prediction).item()
            confidence = torch.max(torch.softmax(prediction, dim=1)).item() * 100
        self.result.config(text=f"{digit}, {confidence:.2f}%")

if __name__ == '__main__':
    # Print device information at the start
    print(f"Using device: {device}")
    
    # Data Augmentation & Normalization
    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load dataset
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    # Split train dataset into train & validation
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    # DataLoader setup
    batch_size = min(4096, len(train_dataset))  # Adjust batch size dynamically
    train_loader = get_dataloader(train_dataset, batch_size, True)
    val_loader = get_dataloader(val_dataset, batch_size, False)
    test_loader = get_dataloader(test_dataset, batch_size, False)

    # Model, Loss, Optimizer
    model = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training Function with Progress Tracking
    def train(epoch):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}', 
                      end='\r')
        
        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch} completed in {epoch_time:.2f} seconds")
        return total_loss / len(train_loader)

    # Validation Function
    def validate():
        model.eval()
        total_loss, correct = 0, 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                total_loss += criterion(output, target).item()
                correct += output.argmax(dim=1).eq(target).sum().item()
        return total_loss / len(val_loader), 100. * correct / len(val_loader.dataset)

    # Test Function
    def test():
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'\nTest set: Average loss: {test_loss:.4f}, '
              f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
        return accuracy

    # Training Loop with Early Stopping
    best_val_loss = float('inf')
    epochs, patience = 20, 3
    best_accuracy = 0

    for epoch in range(1, epochs + 1):
        train_loss = train(epoch)
        val_loss, val_acc = validate()
        test_acc = test()
        
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_acc:.2f}%, Test Acc: {test_acc:.2f}%")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_mnist.pth")
            print("Saved best model")
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping")
                break
        
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            print(f"New best accuracy: {best_accuracy:.2f}%")

    print(f"Training complete. Best accuracy: {best_accuracy:.2f}%")

    # Load Best Model
    model.load_state_dict(torch.load("best_mnist.pth"))
    model.eval()

    # Start GUI
    app = DigitRecognizer()
    app.mainloop() 