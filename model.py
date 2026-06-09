# model.py
# 多视图 SiMVC 模型定义

import torch
import torch.nn as nn
import timm


class MultiViewSiMVC(nn.Module):
    def __init__(self, backbone, n_views=4, n_clusters=8, feat_dim=768, hidden_dim=128):
        super().__init__()
        self.backbone = backbone
        self.n_views = n_views

        self.fusion = nn.Sequential(
            nn.Linear(feat_dim * n_views, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, hidden_dim)
        )

        self.cluster_head = nn.Linear(hidden_dim, n_clusters)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward_features(self, views):
        embs = []
        for v in views:
            feat = self.backbone(v)
            embs.append(feat)
        concat = torch.cat(embs, dim=1)
        z = self.fusion(concat)
        return z

    def forward(self, views):
        z = self.forward_features(views)
        q = torch.softmax(self.cluster_head(z), dim=1)
        logit = self.classifier(z)
        return z, q, logit


def build_model(weight_path=None, device='cpu'):
    """构建模型并加载权重"""
    backbone = timm.create_model(
        'swin_tiny_patch4_window7_224',
        pretrained=False,
        num_classes=0,
        global_pool='avg'
    )

    model = MultiViewSiMVC(backbone, n_views=4, n_clusters=8, feat_dim=768, hidden_dim=128)

    if weight_path:
        state_dict = torch.load(weight_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        print(f"✅ 模型权重已加载: {weight_path}")

    model = model.to(device)
    model.eval()
    return model
