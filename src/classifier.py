import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class SceneClassifier:
    def __init__(self):
        # Using a lightweight, fast model suitable for real-time analysis
        self.weights = models.MobileNet_V3_Small_Weights.DEFAULT
        self.model = models.mobilenet_v3_small(weights=self.weights)
        self.model.eval()
        self.categories = self.weights.meta["categories"]
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict(self, pil_image):
        """Predicts the category of the scene to identify if it is a chart or a real-world environment."""
        img_t = self.transform(pil_image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = self.model(img_t)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
        top_prob, top_catid = torch.topk(probabilities, 1)
        scene_label = self.categories[top_catid[0].item()]
        confidence = top_prob[0].item()
        
        # Mapping ImageNet classes to educational photography profiles
        chart_keywords = ['web site', 'screen', 'monitor', 'slate', 'comic book']
        if any(kw in scene_label.lower() for kw in chart_keywords):
            detected_type = "Standardized Calibration Target / Chart"
            profile = "Chart Suite"
        else:
            detected_type = f"Real-world Scene ({scene_label.title()})"
            profile = "Natural Aesthetics Suite"
            
        return detected_type, profile, confidence