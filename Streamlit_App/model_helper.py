# model_helper.py
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

class_names = ['Front Breakage', 'Front Crushed', 'Front Normal', 'Rear Breakage', 'Rear Crushed', 'Rear Normal','Side Damaged','Side Normal']

cost_estimates = {
    "Maruti Suzuki": {
        'Front Breakage': (15000, 25000),
        'Front Crushed': (8000, 15000),
        'Rear Breakage': (14000, 22000),
        'Rear Crushed': (7000, 13000),
        'Side Damaged': (6000, 16000),
        'Front Normal': (0, 0), 'Rear Normal': (0, 0), 'Side Normal': (0, 0)
    },
    "Hyundai": {
        'Front Breakage': (20000, 32000), 'Front Crushed': (10000, 20000),
        'Rear Breakage': (18000, 30000), 'Rear Crushed': (9000, 18000),
        'Side Damaged': (8000, 18000), 'Front Normal': (0, 0), 'Rear Normal': (0, 0), 'Side Normal': (0, 0)
    },
    "Toyota": {
        'Front Breakage': (30000, 45000), 'Front Crushed': (15000, 28000),
        'Rear Breakage': (25000, 40000), 'Rear Crushed': (14000, 26000),
        'Side Damaged': (12000, 25000), 'Front Normal': (0, 0), 'Rear Normal': (0, 0), 'Side Normal': (0, 0)
    },
    "Tata": {
        'Front Breakage': (18000, 27000), 'Front Crushed': (10000, 18000),
        'Rear Breakage': (16000, 25000), 'Rear Crushed': (9000, 16000),
        'Side Damaged': (7000, 17000), 'Front Normal': (0, 0), 'Rear Normal': (0, 0), 'Side Normal': (0, 0)
    },
    "Mahindra": {
        'Front Breakage': (25000, 35000), 'Front Crushed': (14000, 25000),
        'Rear Breakage': (22000, 32000), 'Rear Crushed': (12000, 21000),
        'Side Damaged': (11000, 22000), 'Front Normal': (0, 0), 'Rear Normal': (0, 0), 'Side Normal': (0, 0)
    },
    "Kia": {
        'Front Breakage': (22000, 32000), 'Front Crushed': (13000, 22000),
        'Rear Breakage': (20000, 30000), 'Rear Crushed': (11000, 20000),
        'Side Damaged': (9000, 19000), 'Front Normal': (0, 0), 'Rear Normal': (0, 0), 'Side Normal': (0, 0)
    },
    "MG": {
        'Front Breakage': (25000, 38000), 'Front Crushed': (14000, 26000),
        'Rear Breakage': (23000, 35000), 'Rear Crushed': (12000, 22000),
        'Side Damaged': (10000, 21000), 'Front Normal': (0, 0), 'Rear Normal': (0, 0), 'Side Normal': (0, 0)
    },
}

# Model
class CarClassifierResNet(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.model = models.resnet50(weights='DEFAULT')
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.layer4.parameters():
            param.requires_grad = True
        self.model.fc = nn.Sequential(
            nn.Dropout(0.29231861632437395),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

trained_model = None

def predict(image_path):
    global trained_model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if trained_model is None:
        trained_model = CarClassifierResNet()
        trained_model.load_state_dict(torch.load("model/saved_model.pth", map_location=device))
        trained_model.to(device)
        trained_model.eval()

    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = trained_model(image_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)[0]
        _, predicted_class = torch.max(output, 1)
        return class_names[predicted_class.item()], probs.cpu().numpy(), class_names

def get_cost_range(brand, damage_type):
    return cost_estimates.get(brand, {}).get(damage_type, (0, 0))