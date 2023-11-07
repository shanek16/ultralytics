import os
import copy
from PIL import Image
import torch
import torch.quantization
from torchvision import datasets, transforms
from sklearn.metrics import mean_squared_error
from torch.utils.data import random_split, Dataset
from torch.quantization import get_default_qconfig, QConfig
from torch.quantization import default_observer, MovingAverageMinMaxObserver


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_names = os.listdir(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image.unsqueeze(0)

# Load your model
model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
model.eval()  # Set the model to evaluation mode
original_model = copy.deepcopy(model)

# Define the transformations to be applied to the images (e.g., resize, normalization)
transform = transforms.Compose([
    transforms.Resize((384, 672)),
    transforms.ToTensor(),
])

# Load the calibration data
calibration_data = CustomImageDataset(img_dir='/workspace/data/safety/calib', transform=transform)
calibration_data = torch.utils.data.Subset(calibration_data, torch.randperm(len(calibration_data)))
calibration_loader, validation_loader = random_split(calibration_data, [len(calibration_data)//2, len(calibration_data)//2])
# Prepare the model for quantization
# Create a custom qconfig
custom_qconfig = QConfig(
    activation=default_observer,  # or replace with your choice of observer
    weight=default_observer  # or replace with your choice of observer
)

# Apply the custom qconfig to ConvTranspose layers
model.qconfig = custom_qconfig
prepared_model = torch.quantization.prepare(model, inplace=False)

# Calibrate the model
for data in calibration_loader:
    with torch.no_grad():
        output = prepared_model(data)

# Convert to quantized model
quantized_model = torch.quantization.convert(prepared_model, inplace=False)
torch.save(quantized_model, 'quantized_midas.pth')

quantized_model.eval()
original_model.eval()
mse = 0
num_samples = 0
try:
    for data, _ in validation_loader:
        with torch.no_grad():
            original_output = original_model(data)
            quantized_output = quantized_model(data)
            mse += mean_squared_error(original_output.cpu().numpy(), quantized_output.cpu().numpy())
            num_samples += 1
except Exception as e:
    print(f"Failed during validation: {e}")
    exit()

# Print the mean squared error over the validation set
print(f'Mean Squared Error on Validation Set: {mse / num_samples}')
