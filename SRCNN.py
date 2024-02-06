import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Resize
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import torch.optim.lr_scheduler as lr_scheduler

# Program Notes:
# - This program implements the SRCNN (Super-Resolution Convolutional Neural Network) model.
# - It uses a learning rate scheduler for automatic learning rate adjustment during training.
# - The resize image size is adjustable and can be set by the user.

# Determine the path to your dataset folder
current_directory = os.getcwd()
# dataset_folder = "gdrive/My Drive/DL_Microsphere3/datasets"
# dataset_path = os.path.join(current_directory, dataset_folder)
dataset_path ="datasets"
# Define a custom dataset class
class SuperResolutionDataset(Dataset):
    def __init__(self, file_list, resize_size):
        self.file_list = file_list
        self.resize_size = resize_size

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        low_res_img_path, high_res_img_path = self.file_list[idx]

        transform = nn.Sequential(
            Resize((self.resize_size, self.resize_size))
        )

        low_res_img = ToTensor()(transform(Image.open(low_res_img_path)))
        high_res_img = ToTensor()(transform(Image.open(high_res_img_path)))

        return low_res_img, high_res_img

# Read the file paths for all images in the dataset
file_list = [(os.path.join(dataset_folder, filename), os.path.join(dataset_folder, filename.replace(".b.nst10000.jpg", ".a.jpg")))
             for filename in os.listdir(dataset_path) if filename.endswith(".b.nst10000.jpg")]
print(f'number of images:{len(file_list)}')


# for file in file_list:
#     print(file)


# Set the seed for reproducibility
torch.manual_seed(42)

# Limit the dataset to a maximum of 3000 samples
file_list = file_list[:3000]

# Split the dataset into training, validation, and test sets
train_val_files, test_files = train_test_split(file_list, test_size=0.1, random_state=42)
train_files, val_files = train_test_split(train_val_files, test_size=0.2, random_state=42)

# Specify the resize sizes
resize_sizes = [128]

for resize_size in resize_sizes:
    print(f"Training with resize_size: {resize_size}")

    # Create the datasets
    train_dataset = SuperResolutionDataset(train_files, resize_size)
    val_dataset = SuperResolutionDataset(val_files, resize_size)
    test_dataset = SuperResolutionDataset(test_files, resize_size)

    # Create the data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Define the SRCNN model
    class SRCNN(nn.Module):
        def __init__(self, in_channels, out_channels):
            super(SRCNN, self).__init__()
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4)
            self.conv2 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0)
            self.conv3 = nn.Conv2d(32, out_channels, kernel_size=5, stride=1, padding=2)
            self.relu = nn.ReLU()

        def forward(self, x):
            out = self.relu(self.conv1(x))
            out = self.relu(self.conv2(out))
            out = self.conv3(out)
            return out

    # Create an instance of the SRCNN model
    model = SRCNN(1, 1)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Define the learning rate scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    # Train the model
    num_epochs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        train_loss = 0.0
        val_loss = 0.0

        # Training phase
        model.train()
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation phase
        model.eval()
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()

        # Compute average losses
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        # Adjust learning rate
        scheduler.step(val_loss)

        # Save the model at epoch 10 and epoch 20
        if epoch == 9:
            torch.save(model.state_dict(), f"super_resolution_model_SRCNN_nst10000_{resize_size}_EP10.pth")
        if epoch == 19:
            torch.save(model.state_dict(), f"super_resolution_model_SRCNN_nst10000_{resize_size}_EP20.pth")

        # Print epoch progress
        print(f"Epoch {epoch+1}/{num_epochs} Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")

    # Save the final trained model
    # Save SRCNN model
    #torch.save(model.state_dict(), "srcnn_model.pth")
    model_filename = f"super_resolution_model_SRCNN_nst10000_{resize_size}_EP{num_epochs}.pth"
    torch.save(model.state_dict(), model_filename)
    print(f"Training with resize_size: {resize_size} completed!\n")