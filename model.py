import torch
import timm
from torchvision import transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Classes dans le même ordre que pendant l'entraînement
class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']


def load_model(pth_path="best_model_new.pth"):
    model = timm.create_model('efficientnet_b3', pretrained=False, num_classes=7)
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# Transform exactement identique à l'entraînement
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def predict_image(image_path, model, threshold=0.59):
    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)[0]
        confidence, predicted = torch.max(probs, 0)

    pred_class = class_names[predicted.item()]
    confidence = confidence.item()

    if confidence >= threshold:
        return pred_class, confidence, "Confident"
    else:
        return pred_class, confidence, "Incertain → À vérifier par un médecin"