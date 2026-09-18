import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

class FeatureExtractor:
    def __init__(self):
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)

        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract(self, image_path_or_file):
        img = Image.open(image_path_or_file).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0)

        with torch.no_grad():
            feature_vector = self.model(img_tensor)

        return feature_vector.numpy().flatten()

if __name__ == "__main__":
    extractor = FeatureExtractor()
    print("модель загружена!")

