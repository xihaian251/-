# inference.py
# 推理逻辑

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from model import build_model

# 图像预处理
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def preprocess_image(image_path):
    """加载并预处理单张图片"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_tensor = transforms.ToTensor()(img)
    img_normalized = normalize(img_tensor).unsqueeze(0)
    return img, img_normalized

class BreastCancerPredictor:
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        self.model = build_model(model_path, device=self.device)

        # 注册 Grad-CAM hook
        self.target_layer = self.model.backbone.layers[-1].blocks[-1]
        self.features = []
        self.gradients = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.features.append(output)
        def backward_hook(module, grad_in, grad_out):
            self.gradients.append(grad_out[0])
        self.handle_f = self.target_layer.register_forward_hook(forward_hook)
        self.handle_b = self.target_layer.register_full_backward_hook(backward_hook)

    def predict(self, image_path):
        """对单张图片做推理"""
        img_pil, img_tensor = preprocess_image(image_path)
        img_tensor = img_tensor.to(self.device)

        # 前向传播（单视图复制4份）
        with torch.no_grad():
            views = [img_tensor] * 4
            _, _, logit = self.model(views)
            malignant_prob = torch.sigmoid(logit).item()

        prediction = '恶性' if malignant_prob > 0.5 else '良性'
        confidence = malignant_prob if malignant_prob > 0.5 else 1 - malignant_prob

        # Grad-CAM
        heatmap = self._compute_heatmap(img_tensor)

        return {
            'prediction': prediction,
            'confidence': confidence,
            'malignant_prob': malignant_prob,
            'heatmap': heatmap,
            'image': img_pil
        }

    def _compute_heatmap(self, img_tensor):
        self.features.clear()
        self.gradients.clear()

        with torch.enable_grad():
            views = [img_tensor.to(self.device).requires_grad_(True)] * 4
            _, _, logit = self.model(views)

        self.model.zero_grad()
        logit[0, 0].backward(retain_graph=True)

        A = self.features[-1]
        grad = self.gradients[-1]

        if len(A.shape) == 3:
            B, L, C = A.shape
            H = W = int(L ** 0.5)
            A = A.reshape(B, H, W, C).permute(0, 3, 1, 2)
        elif len(A.shape) == 4:
            A = A.permute(0, 3, 1, 2)

        if len(grad.shape) == 3:
            B, L, C = grad.shape
            H = W = int(L ** 0.5)
            grad = grad.reshape(B, H, W, C).permute(0, 3, 1, 2)
        elif len(grad.shape) == 4:
            grad = grad.permute(0, 3, 1, 2)

        if A.shape[1] != grad.shape[1]:
            min_c = min(A.shape[1], grad.shape[1])
            A = A[:, :min_c, :, :]
            grad = grad[:, :min_c, :, :]

        weights = torch.mean(grad, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * A, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().detach().numpy()

        if cam.max() - cam.min() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam

    def overlay_heatmap(self, image, heatmap, alpha=0.5):
        img = image.resize((224, 224))
        img = np.array(img)
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        return cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
