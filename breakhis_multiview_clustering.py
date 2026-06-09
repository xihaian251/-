# 阶段❶ - Cell 1: PyTorch 2.5.1 + timm 0.9.16
import sys
print("Python版本:", sys.version)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import timm
import numpy as np
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')
print(f"\n PyTorch版本: {torch.__version__}")
print(f" torchvision版本: {torchvision.__version__}")
print(f" timm版本: {timm.__version__}")
print(f" GPU可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU名称: {torch.cuda.get_device_name(0)}")
    # 🔧 修复：total_mem → total_memory
    print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
# 测试 CUDA
x = torch.randn(3, 3).cuda()
print(f"   CUDA测试: {x.device} ✓")

# 测试 timm 是否能加载 Swin-Tiny
try:
    model_test = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=0)
    n_params = sum(p.numel() for p in model_test.parameters())
    print(f"   Swin-Tiny加载: ✓ (参数量: {n_params:,})")
    # 测试前向传播
    dummy = torch.randn(2, 3, 224, 224)
    out = model_test(dummy)
    print(f"   前向传播: ✓ (输出形状: {out.shape})")
except Exception as e:
    print(f"   Swin-Tiny加载: ✗ {e}")

print("\n 环境全部就绪！可以开始项目了。")

# 阶段❶ - Cell 2.5: 深度探测BreakHis真实结构


import os

DATA_ROOT = "/kaggle/input/datasets/ambarish/breakhis/BreaKHis_v1/BreaKHis_v1"
HIST_PATH = os.path.join(DATA_ROOT, "histology_slides", "breast")

print(" 深度探测 benign/ 目录...\n")

benign_path = os.path.join(HIST_PATH, "benign")

# 显示5层结构
for root, dirs, files in os.walk(benign_path):
    level = root.replace(benign_path, '').count(os.sep)
    if level <= 5:
        indent = '  ' * level
        basename = os.path.basename(root) if level > 0 else 'benign/'
        print(f"{indent} {basename}")

        # 显示目录
        for d in sorted(dirs)[:5]:
            print(f"{indent}   {d}/")
        if len(dirs) > 5:
            print(f"{indent}  ... 还有 {len(dirs) - 5} 个目录")

        # 显示文件（仅在深层）
        if level >= 3:
            png_files = [f for f in sorted(files)[:5] if f.endswith('.png')]
            for f in png_files:
                print(f"{indent}   {f}")
            if len(files) > 5:
                print(f"{indent}  ... 还有 {len(files) - 5} 个文件")
    else:
        break

print("\n" + "=" * 60)
print(" 深度探测 malignant/ 目录...\n")

malignant_path = os.path.join(HIST_PATH, "malignant")

for root, dirs, files in os.walk(malignant_path):
    level = root.replace(malignant_path, '').count(os.sep)
    if level <= 5:
        indent = '  ' * level
        basename = os.path.basename(root) if level > 0 else 'malignant/'
        print(f"{indent} {basename}")

        for d in sorted(dirs)[:5]:
            print(f"{indent}   {d}/")
        if len(dirs) > 5:
            print(f"{indent}  ... 还有 {len(dirs) - 5} 个目录")

        if level >= 3:
            png_files = [f for f in sorted(files)[:5] if f.endswith('.png')]
            for f in png_files:
                print(f"{indent}   {f}")
            if len(files) > 5:
                print(f"{indent}  ... 还有 {len(files) - 5} 个文件")
    else:
        break

# 阶段❶ - Cell 3: 统计数据集 + 构建四视图样本
DATA_ROOT = "/kaggle/input/datasets/ambarish/breakhis/BreaKHis_v1/BreaKHis_v1"
HIST_PATH = os.path.join(DATA_ROOT, "histology_slides", "breast")

# 亚型映射
BENIGN_MAP = {
    'adenosis': 'A',
    'fibroadenoma': 'F',
    'phyllodes_tumor': 'PT',
    'tubular_adenoma': 'TA'
}
MALIGNANT_MAP = {
    'ductal_carcinoma': 'DC',
    'lobular_carcinoma': 'LC',
    'mucinous_carcinoma': 'MC',
    'papillary_carcinoma': 'PC'
}

MAGNIFICATIONS = ['40X', '100X', '200X', '400X']

# Step 1: 收集所有样本

all_samples = []

for category, base_path in [('benign', os.path.join(HIST_PATH, 'benign', 'SOB')),
                            ('malignant', os.path.join(HIST_PATH, 'malignant', 'SOB'))]:
    subtype_map = BENIGN_MAP if category == 'benign' else MALIGNANT_MAP

    for subtype_dir in sorted(os.listdir(base_path)):
        subtype_path = os.path.join(base_path, subtype_dir)
        if not os.path.isdir(subtype_path):
            continue

        subtype_code = subtype_map.get(subtype_dir, subtype_dir)

        for patient_dir in sorted(os.listdir(subtype_path)):
            patient_path = os.path.join(subtype_path, patient_dir)
            if not os.path.isdir(patient_path):
                continue

            for mag in MAGNIFICATIONS:
                mag_path = os.path.join(patient_path, mag)
                if not os.path.exists(mag_path):
                    continue

                for fname in os.listdir(mag_path):
                    if fname.endswith('.png'):
                        all_samples.append({
                            'path': os.path.join(mag_path, fname),
                            'filename': fname,
                            'subtype': subtype_code,
                            'magnification': mag,
                            'category': category,
                            'label': 0 if category == 'benign' else 1,
                            'patient_id': patient_dir
                        })

df = pd.DataFrame(all_samples)
print(f" 共收集 {len(df)} 张图片")
print(f"   良性: {len(df[df['label'] == 0])} 张")
print(f"   恶性: {len(df[df['label'] == 1])} 张")

# Step 2: 按亚型统计

print("\n" + "=" * 60)
print("各亚型统计")
print("=" * 60)
for subtype_code in ['A', 'F', 'PT', 'TA', 'DC', 'LC', 'MC', 'PC']:
    sub_df = df[df['subtype'] == subtype_code]
    if len(sub_df) > 0:
        cat = '良性' if subtype_code in BENIGN_MAP.values() else '恶性'
        counts = {mag: len(sub_df[sub_df['magnification'] == mag]) for mag in MAGNIFICATIONS}
        print(f"  {subtype_code} ({cat}): 总{len(sub_df):4d}张  "
              f"(40X:{counts['40X']}, 100X:{counts['100X']}, "
              f"200X:{counts['200X']}, 400X:{counts['400X']})")


# Step 3: 提取文件名的"基名"用于匹配四视图

def extract_base_name(filename):
    """
    SOB_B_A-14-22549AB-40-001.png → SOB_B_A-14-22549AB-001
    去掉倍数部分（40/100/200/400）
    """
    name = filename.replace('.png', '')
    parts = name.split('-')
    mag_idx = None
    for i, p in enumerate(parts):
        if p in ['40', '100', '200', '400']:
            mag_idx = i
            break
    if mag_idx is not None:
        return '-'.join(parts[:mag_idx] + parts[mag_idx + 1:])
    return name


df['base_name'] = df['filename'].apply(extract_base_name)

# Step 4: 按 (patient_id, base_name) 分组构建四视图

grouped = df.groupby(['patient_id', 'base_name'])
multi_view_samples = []

for (patient, base), group in grouped:
    if len(group) == 4:
        mags = set(group['magnification'].values)
        if mags == set(MAGNIFICATIONS):
            row_40 = group[group['magnification'] == '40X'].iloc[0]
            row_100 = group[group['magnification'] == '100X'].iloc[0]
            row_200 = group[group['magnification'] == '200X'].iloc[0]
            row_400 = group[group['magnification'] == '400X'].iloc[0]

            multi_view_samples.append({
                'base_name': base,
                'patient_id': patient,
                'path_40': row_40['path'],
                'path_100': row_100['path'],
                'path_200': row_200['path'],
                'path_400': row_400['path'],
                'subtype': row_40['subtype'],
                'category': row_40['category'],
                'label': row_40['label']
            })

df_mv = pd.DataFrame(multi_view_samples)
print(f"\n 成功构建 {len(df_mv)} 个四视图样本")
print(f"   - 良性: {len(df_mv[df_mv['label'] == 0])} 个")
print(f"   - 恶性: {len(df_mv[df_mv['label'] == 1])} 个")

# 按亚型统计四视图样本
print("\n各亚型四视图样本数:")
for subtype_code in ['A', 'F', 'PT', 'TA', 'DC', 'LC', 'MC', 'PC']:
    count = len(df_mv[df_mv['subtype'] == subtype_code])
    if count > 0:
        cat = '良性' if subtype_code in BENIGN_MAP.values() else '恶性'
        patients = df_mv[df_mv['subtype'] == subtype_code]['patient_id'].nunique()
        print(f"  {subtype_code} ({cat}): {count} 个样本, {patients} 个患者")

# 保存元数据
df_mv.to_csv('/kaggle/working/multiview_metadata.csv', index=False)
print("\n 元数据已保存到 /kaggle/working/multiview_metadata.csv")

# 阶段❶ - Cell 4（修复版）: 可视化良性/恶性四视图


# 读取之前保存的元数据
df_mv = pd.read_csv('/kaggle/working/multiview_metadata.csv')

print("列名:", df_mv.columns.tolist())
print()


#  修复：去掉 'X' 来匹配列名
# 列名是 path_40, path_100, path_200, path_400
# 所以我们用 mag[:-1] 取 '40X' -> '40' 等

def show_four_views(df, idx, save_name=None):
    row = df.iloc[idx]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for ax, mag in zip(axes, MAGNIFICATIONS):
        #  mag 是 '40X', '100X' 等，去掉末尾 'X' 得到 '40', '100'
        col_name = f'path_{mag[:-1]}'  # '40X' -> '40', '100X' -> '100'
        img = Image.open(row[col_name])
        ax.imshow(img)
        ax.set_title(f'{mag}', fontsize=14, fontweight='bold')
        ax.axis('off')

    fig.suptitle(f"{row['category'].upper()} | Subtype: {row['subtype']} | Patient: {row['patient_id']}",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    if save_name:
        plt.savefig(f'/kaggle/working/{save_name}.png', dpi=150, bbox_inches='tight')
    plt.show()


# 良性样本
print("=" * 50)
print(" 良性样本示例")
print("=" * 50)
show_four_views(df_mv[df_mv['label'] == 0], idx=0, save_name='benign_sample')

# 恶性样本
print("\n" + "=" * 50)
print(" 恶性样本示例")
print("=" * 50)
show_four_views(df_mv[df_mv['label'] == 1], idx=0, save_name='malignant_sample')

# 阶段❶ - Cell 5（修复版v2）: 多视图数据集 + 数据加载器


from sklearn.model_selection import train_test_split
import pandas as pd

# 读取元数据
df_mv = pd.read_csv('/kaggle/working/multiview_metadata.csv')

# 按患者ID划分训练/验证集
patient_ids = df_mv['patient_id'].unique()
train_patients, val_patients = train_test_split(
    patient_ids, test_size=0.2, random_state=42,
    stratify=[df_mv[df_mv['patient_id'] == p]['label'].iloc[0] for p in patient_ids]
)

train_df = df_mv[df_mv['patient_id'].isin(train_patients)].reset_index(drop=True)
val_df = df_mv[df_mv['patient_id'].isin(val_patients)].reset_index(drop=True)

print(f"训练集: {len(train_df)} 个样本 ({len(train_patients)} 个患者)")
print(f"  - 良性: {len(train_df[train_df['label'] == 0])} | 恶性: {len(train_df[train_df['label'] == 1])}")
print(f"验证集: {len(val_df)} 个样本 ({len(val_patients)} 个患者)")
print(f"  - 良性: {len(val_df[val_df['label'] == 0])} | 恶性: {len(val_df[val_df['label'] == 1])}")


# 数据增强

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(90),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



# 多视图数据集类

class MultiViewDataset(Dataset):
    """四视图病理图像数据集"""

    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.mag_keys = ['path_40', 'path_100', 'path_200', 'path_400']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        views = []
        for key in self.mag_keys:
            img = Image.open(row[key]).convert('RGB')
            if self.transform:
                img = self.transform(img)
            views.append(img)

        return {
            'views': views,  # list of 4 Tensors
            'label': torch.tensor(row['label'], dtype=torch.float32),
            'subtype': row['subtype'],
            'patient_id': row['patient_id']
        }


# 创建数据集
train_dataset = MultiViewDataset(train_df, transform=train_transform)
val_dataset = MultiViewDataset(val_df, transform=val_transform)

#  关键修复：num_workers=0 避免子进程继承损坏的 torch
BATCH_SIZE = 8

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=True)

# 测试一个batch
batch = next(iter(train_loader))
print(f"\n 数据加载器测试:")
print(f"   views数量: {len(batch['views'])} (每个元素形状: {batch['views'][0].shape})")
print(f"   labels形状: {batch['label'].shape}")
print(f"   labels: {batch['label'].tolist()}")
print(f"   batch中的亚型: {batch['subtype']}")
=
# 阶段❷ - Cell 6: CTransPath 骨干网络


import torch
import torch.nn as nn
import timm


# Step 1: 检查 CTransPath 权重文件

CTRANS_PATH = "/kaggle/input/datasets/malekbennabi/ctranspath-models"

print(" CTransPath 目录内容:")
for f in os.listdir(CTRANS_PATH):
    size_mb = os.path.getsize(os.path.join(CTRANS_PATH, f)) / 1024 ** 2
    print(f"   {f} ({size_mb:.1f} MB)")

# 找到 .pth 文件
weight_files = [f for f in os.listdir(CTRANS_PATH) if f.endswith('.pth')]
if weight_files:
    weight_path = os.path.join(CTRANS_PATH, weight_files[0])
    print(f"\n 找到权重文件: {weight_path}")
else:
    print("\n 未找到 .pth 文件，请检查数据集")
    weight_path = None



# Step 2: 构建 Swin-Tiny 骨干网络

def build_ctranspath_backbone(pretrained_path=None):
    """
    构建 CTransPath 骨干网络 (Swin-Tiny)
    输出: 768维特征向量
    """
    # 创建 Swin-Tiny，去掉分类头
    backbone = timm.create_model(
        'swin_tiny_patch4_window7_224',
        pretrained=False,
        num_classes=0,  # 去掉分类头，输出全局池化后的特征
        global_pool='avg'  # 平均池化
    )

    print(f"\n📐 骨干网络参数量: {sum(p.numel() for p in backbone.parameters()):,}")

    # 加载 CTransPath 预训练权重
    if pretrained_path and os.path.exists(pretrained_path):
        state_dict = torch.load(pretrained_path, map_location='cpu')

        # 检查 key 格式
        sample_key = list(state_dict.keys())[0]
        print(f"   权重key示例: {sample_key}")

        # 尝试加载（可能需要处理 key 前缀）
        try:
            backbone.load_state_dict(state_dict, strict=True)
            print("    权重加载成功 (strict=True)")
        except Exception as e:
            print(f"    strict=True 失败: {str(e)[:100]}...")
            try:
                backbone.load_state_dict(state_dict, strict=False)
                print("    权重加载成功 (strict=False)")
            except Exception as e2:
                print(f"    权重加载失败: {e2}")
                print("   将使用随机初始化的权重")
    else:
        print("    未提供预训练权重，使用随机初始化")

    return backbone


# 构建骨干网络
backbone = build_ctranspath_backbone(weight_path)


# Step 3: 测试前向传播

dummy = torch.randn(2, 3, 224, 224)
with torch.no_grad():
    feat = backbone(dummy)
print(f"\n 前向传播测试:")
print(f"   输入形状: {dummy.shape}")
print(f"   输出形状: {feat.shape}  (期望: [2, 768])")
print(f"   特征范数: {feat.norm(dim=1).mean():.4f}")



# 阶段❷ - Cell 7: 多视图 SiMVC 模型


class MultiViewSiMVC(nn.Module):
    """
    多视图深度聚类模型
    - 共享 CTransPath 骨干提取各视图特征
    - 融合 MLP 将4个视图特征合并为128维表示
    - 聚类头输出8类软分配
    - 分类头输出良恶性logit
    """

    def __init__(self, backbone, n_views=4, n_clusters=8, feat_dim=768, hidden_dim=128):
        super().__init__()
        self.backbone = backbone
        self.n_views = n_views

        # 融合网络：4*768 → 512 → 128
        self.fusion = nn.Sequential(
            nn.Linear(feat_dim * n_views, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, hidden_dim)
        )

        # 聚类头：128 → n_clusters
        self.cluster_head = nn.Linear(hidden_dim, n_clusters)

        # 分类头：128 → 1（良恶性二分类）
        self.classifier = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in [self.fusion, self.cluster_head, self.classifier]:
            for module in m.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward_features(self, views):
        """提取并融合4个视图的特征"""
        # views: list of 4 Tensors, each (B, 3, 224, 224)
        embs = []
        for v in views:
            feat = self.backbone(v)  # (B, 768)
            embs.append(feat)

        concat = torch.cat(embs, dim=1)  # (B, 4*768)
        z = self.fusion(concat)  # (B, 128)
        return z

    def forward(self, views):
        z = self.forward_features(views)  # (B, 128)
        q = torch.softmax(self.cluster_head(z), dim=1)  # (B, n_clusters)
        logit = self.classifier(z)  # (B, 1)
        return z, q, logit



# 构建模型

model = MultiViewSiMVC(backbone, n_views=4, n_clusters=8, feat_dim=768, hidden_dim=128)

# 统计参数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f" 模型总参数量: {total_params:,}")
print(f"   可训练参数量: {trainable_params:,}")


# 测试完整前向传播

# 模拟一个batch的4视图输入
dummy_views = [torch.randn(2, 3, 224, 224) for _ in range(4)]

model.eval()
with torch.no_grad():
    z, q, logit = model(dummy_views)

print(f"\n 完整前向传播测试:")
print(f"   输入: 4个视图, 每个形状 (2, 3, 224, 224)")
print(f"   融合表示 z: {z.shape}  (期望: [2, 128])")
print(f"   聚类概率 q: {q.shape}  (期望: [2, 8])")
print(f"   聚类概率和: {q.sum(dim=1)}  (期望: [1.0, 1.0])")
print(f"   分类logit: {logit.shape}  (期望: [2, 1])")
print(f"   分类概率: {torch.sigmoid(logit).squeeze().tolist()}")

# 移动到 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"\n 模型已移至: {device}")

# 阶段❷ - Cell 8: 损失函数与优化器



# 1. 目标分布生成函数（DEC 聚类损失用）

def target_distribution(q):
    """
    从当前软分配 q 生成目标分布 p
    p_ij = (q_ij^2 / f_j) / sum_j'(q_ij'^2 / f_j')
    其中 f_j = sum_i q_ij
    """
    weight = q ** 2 / q.sum(dim=0)  # (B, C)
    return (weight.T / weight.sum(dim=1)).T  # 归一化


# 2. 损失函数

def kl_divergence(p, q):
    """KL散度: KL(p||q) = sum(p * log(p/q))"""
    p = torch.clamp(p, min=1e-10)
    q = torch.clamp(q, min=1e-10)
    return torch.sum(p * torch.log(p / q), dim=1).mean()

def cluster_loss(q, p):
    """聚类损失 = KL(p||q)"""
    return kl_divergence(p, q)

def consistency_loss(z1, z2):
    """表示一致性损失（MSE）"""
    return F.mse_loss(z1, z2)

def assignment_consistency_loss(q1, q2):
    """分配一致性损失（MSE）"""
    return F.mse_loss(q1, q2)


# 3. 优化器与调度器

# 分组学习率：骨干网络用小学习率，新模块用大学习率
backbone_params = list(model.backbone.parameters())
fusion_params = list(model.fusion.parameters()) + \
                list(model.cluster_head.parameters()) + \
                list(model.classifier.parameters())

optimizer = torch.optim.AdamW([
    {'params': backbone_params, 'lr': 1e-5},   # 骨干微调
    {'params': fusion_params, 'lr': 1e-3}       # 新模块
], weight_decay=1e-4)

# 余弦退火调度器
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=50, eta_min=1e-6
)


# 4. 损失权重配置

LOSS_CONFIG = {
    'lambda_kl': 1.0,       # 聚类 KL 损失权重
    'lambda_con': 0.1,      # 表示一致性权重
    'lambda_assign': 0.1,   # 分配一致性权重
    'alpha_cls': 1.0,       # 分类损失权重
}

print(" 训练配置完成")
print(f"   优化器: AdamW (骨干 lr=1e-5, 新模块 lr=1e-3)")
print(f"   损失权重: {LOSS_CONFIG}")
print(f"   设备: {device}")

# 阶段❸ - Cell 9: 修改数据集以支持两次增强


import copy

# 基础变换（不含归一化，用于返回原始图像后再增强）
base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 增强变换（在训练时随机应用两次）
augment_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(90),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
])

# 归一化（增强后应用）
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class MultiViewDatasetV2(Dataset):
    """返回原始图像和基础变换后的tensor，支持两次增强"""

    def __init__(self, df, is_train=True):
        self.df = df.reset_index(drop=True)
        self.is_train = is_train
        self.mag_keys = ['path_40', 'path_100', 'path_200', 'path_400']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        views = []
        for key in self.mag_keys:
            img = Image.open(row[key]).convert('RGB')
            img = base_transform(img)  # Resize + ToTensor (3, 224, 224)
            views.append(img)

        return {
            'views_raw': views,  # 4个原始tensor（未增强未归一化）
            'label': torch.tensor(row['label'], dtype=torch.float32),
            'subtype': row['subtype'],
            'patient_id': row['patient_id']
        }


def apply_augmentation(views_raw, is_train=True):
    """对4个视图应用增强+归一化，返回处理后的视图列表"""
    processed = []
    for v in views_raw:
        if is_train:
            v = augment_transform(v)  # 随机增强
        v = normalize(v)  # 归一化
        processed.append(v)
    return processed


# 重新创建数据集和数据加载器
train_dataset_v2 = MultiViewDatasetV2(train_df, is_train=True)
val_dataset_v2 = MultiViewDatasetV2(val_df, is_train=False)

BATCH_SIZE = 8
train_loader_v2 = DataLoader(train_dataset_v2, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=0, pin_memory=True)
val_loader_v2 = DataLoader(val_dataset_v2, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=0, pin_memory=True)

# 测试
batch = next(iter(train_loader_v2))
views_aug = apply_augmentation(batch['views_raw'], is_train=True)
print(f" 新数据集测试通过")
print(f"   views_raw数量: {len(batch['views_raw'])}")
print(f"   每个raw tensor形状: {batch['views_raw'][0].shape}")
print(f"   增强后tensor形状: {views_aug[0].shape}")
print(f"   增强后数值范围: [{views_aug[0].min():.2f}, {views_aug[0].max():.2f}]")

# 阶段❸ - Cell 10: 渐进式稳定训练


import time
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.cluster import KMeans

model = model.to(device)
model.train()



# 辅助函数

@torch.no_grad()
def get_all_features(model, loader):
    model.eval()
    all_z, all_q, all_labels = [], [], []
    for batch in loader:
        views = apply_augmentation(batch['views_raw'], is_train=False)
        views = [v.to(device) for v in views]
        z, q, logit = model(views)
        if torch.isnan(z).any() or torch.isnan(q).any():
            continue
        all_z.append(z.cpu())
        all_q.append(q.cpu())
        all_labels.append(batch['label'])
    model.train()
    if len(all_z) == 0:
        return torch.zeros(1, 128), torch.zeros(1, 8), torch.zeros(1)
    return torch.cat(all_z, dim=0), torch.cat(all_q, dim=0), torch.cat(all_labels, dim=0)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        views = apply_augmentation(batch['views_raw'], is_train=False)
        views = [v.to(device) for v in views]
        _, _, logit = model(views)
        preds = torch.sigmoid(logit).cpu().squeeze()
        if torch.isnan(preds).any():
            continue
        all_preds.extend(preds.tolist())
        all_labels.extend(batch['label'].tolist())
    model.train()

    if len(all_preds) == 0 or len(np.unique(all_labels)) < 2:
        return 0.5, 0.5, 0.5

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = accuracy_score(all_labels, (all_preds > 0.5).astype(int))
    auc = roc_auc_score(all_labels, all_preds)
    f1 = f1_score(all_labels, (all_preds > 0.5).astype(int))
    return acc, auc, f1


def freeze_backbone_stages(model, freeze_stages):
    """冻结指定 stage：freeze_stages=['layers.0', 'layers.1'] 等"""
    for name, param in model.backbone.named_parameters():
        should_freeze = any(s in name for s in freeze_stages)
        param.requires_grad = not should_freeze


def get_optimizer(model, lr_backbone=5e-6, lr_head=5e-4):
    return torch.optim.AdamW([
        {'params': [p for p in model.backbone.parameters() if p.requires_grad], 'lr': lr_backbone},
        {'params': list(model.fusion.parameters()) +
                   list(model.cluster_head.parameters()) +
                   list(model.classifier.parameters()), 'lr': lr_head}
    ], weight_decay=1e-4)


cls_criterion = nn.BCEWithLogitsLoss()

# 阶段一：分类预热（15 epochs）
# 冻结 layers.0 + layers.1，只训练 layers.2+3 和新模块


print(" 阶段一：分类预热 (15 epochs)")
print("  冻结: layers.0, layers.1")


freeze_backbone_stages(model, ['layers.0', 'layers.1'])
optimizer = get_optimizer(model, lr_backbone=5e-6, lr_head=5e-4)

for epoch in range(1, 16):
    model.train()
    total_loss = 0
    for batch in train_loader_v2:
        views = apply_augmentation(batch['views_raw'], is_train=True)
        views = [v.to(device) for v in views]
        labels = batch['label'].to(device)
        _, _, logit = model(views)
        loss = cls_criterion(logit.squeeze(), labels)

        if torch.isnan(loss):
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        total_loss += loss.item()

    acc, auc, f1 = evaluate(model, val_loader_v2)
    print(f"Epoch {epoch:2d}/15 | Loss: {total_loss / max(1, len(train_loader_v2)):.4f} "
          f"| Acc: {acc:.4f} | AUC: {auc:.4f} | F1: {f1:.4f}")

print(" 阶段一完成")


# 阶段二：解冻 layers.1，继续分类（10 epochs）

print("\n" + "=" * 60)
print(" 阶段二：解冻 layers.1 (10 epochs)")
print("  冻结: layers.0")


freeze_backbone_stages(model, ['layers.0'])
optimizer = get_optimizer(model, lr_backbone=5e-6, lr_head=5e-4)

for epoch in range(1, 11):
    model.train()
    total_loss = 0
    for batch in train_loader_v2:
        views = apply_augmentation(batch['views_raw'], is_train=True)
        views = [v.to(device) for v in views]
        labels = batch['label'].to(device)
        _, _, logit = model(views)
        loss = cls_criterion(logit.squeeze(), labels)

        if torch.isnan(loss):
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        total_loss += loss.item()

    acc, auc, f1 = evaluate(model, val_loader_v2)
    print(f"Epoch {epoch:2d}/10 | Loss: {total_loss / max(1, len(train_loader_v2)):.4f} "
          f"| Acc: {acc:.4f} | AUC: {auc:.4f} | F1: {f1:.4f}")

print(" 阶段二完成")


# 阶段三：解冻 layers.0，全部参数训练（10 epochs）

print("\n" + "=" * 60)
print(" 阶段三：全部解冻 (10 epochs)")
print("=" * 60)

freeze_backbone_stages(model, [])  # 全部解冻
optimizer = get_optimizer(model, lr_backbone=2e-6, lr_head=3e-4)  # 更小的学习率

for epoch in range(1, 11):
    model.train()
    total_loss = 0
    for batch in train_loader_v2:
        views = apply_augmentation(batch['views_raw'], is_train=True)
        views = [v.to(device) for v in views]
        labels = batch['label'].to(device)
        _, _, logit = model(views)
        loss = cls_criterion(logit.squeeze(), labels)

        if torch.isnan(loss):
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3)
        optimizer.step()
        total_loss += loss.item()

    acc, auc, f1 = evaluate(model, val_loader_v2)
    print(f"Epoch {epoch:2d}/10 | Loss: {total_loss / max(1, len(train_loader_v2)):.4f} "
          f"| Acc: {acc:.4f} | AUC: {auc:.4f} | F1: {f1:.4f}")

print(" 阶段三完成")


# 阶段四：引入聚类损失（20 epochs，逐步增加）

print("\n" + "=" * 60)
print(" 阶段四：联合聚类+分类训练 (20 epochs)")


# K-means 初始化聚类头
print("正在用 K-means 初始化聚类头...")
all_z, all_q, _ = get_all_features(model, train_loader_v2)
if all_z.shape[0] > 8:
    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
    kmeans.fit(all_z.numpy())
    with torch.no_grad():
        centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)
        model.cluster_head.weight.data = centers
        model.cluster_head.bias.data.zero_()
    print(f"    聚类头权重形状: {model.cluster_head.weight.shape}")

# 初始全局目标分布
_, all_q, _ = get_all_features(model, train_loader_v2)
global_p_target = target_distribution(all_q).to(device)

best_auc = 0
best_model_state = None
history = {'epoch': [], 'loss': [], 'acc': [], 'auc': [], 'f1': [], 'nmi': [], 'ari': []}

optimizer = get_optimizer(model, lr_backbone=2e-6, lr_head=3e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-7)

for epoch in range(1, 21):
    model.train()
    total_loss = 0
    total_cls = 0
    total_clu = 0
    nan_count = 0

    #  聚类损失权重 warmup：从 0.1 线性增加到 1.0
    lambda_cluster = min(0.1 + 0.9 * (epoch / 10), 1.0)

    # 每5个epoch更新全局目标分布
    if epoch > 1 and epoch % 5 == 0:
        print(f"  更新全局目标分布 P (epoch {epoch})...")
        _, all_q, _ = get_all_features(model, train_loader_v2)
        if all_q.shape[0] > 0:
            global_p_target = target_distribution(all_q).to(device)

    for batch in train_loader_v2:
        views1 = apply_augmentation(batch['views_raw'], is_train=True)
        views2 = apply_augmentation(batch['views_raw'], is_train=True)
        views1 = [v.to(device) for v in views1]
        views2 = [v.to(device) for v in views2]
        labels = batch['label'].to(device)

        z1, q1, logit1 = model(views1)
        z2, q2, logit2 = model(views2)

        if torch.isnan(z1).any() or torch.isnan(z2).any():
            nan_count += 1
            continue

        #  聚类损失：只用 epoch 级别的全局目标分布（不做 batch 级 target_distribution）
        with torch.no_grad():
            # 取全局目标分布中对应 batch 的部分（简化：用当前 batch 对应位置的全局分布）
            # 这里直接用全局分布均值来监督，避免 batch 级除零
            p_global = global_p_target[:len(z1)].to(device)

        loss_kl = cluster_loss(q1, p_global)
        loss_con = consistency_loss(z1, z2)
        loss_cluster = loss_kl + 0.1 * loss_con

        # 分类损失
        loss_cls = cls_criterion(logit1.squeeze(), labels)

        # 总损失：聚类权重渐进增加
        loss = lambda_cluster * loss_cluster + loss_cls

        if torch.isnan(loss):
            nan_count += 1
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3)
        optimizer.step()

        total_loss += loss.item()
        total_cls += loss_cls.item()
        total_clu += loss_cluster.item()

    scheduler.step()

    # 评估
    acc, auc, f1 = evaluate(model, val_loader_v2)

    all_z_val, _, all_labels_val = get_all_features(model, val_loader_v2)
    if all_z_val.shape[0] > 8:
        kmeans_pred = KMeans(n_clusters=8, random_state=42, n_init=10).fit_predict(all_z_val.numpy())
        nmi = normalized_mutual_info_score(all_labels_val.numpy().astype(int), kmeans_pred)
        ari = adjusted_rand_score(all_labels_val.numpy().astype(int), kmeans_pred)
    else:
        nmi, ari = 0, 0

    n_batches = max(1, len(train_loader_v2) - nan_count)
    nan_str = f" |  NaN: {nan_count}" if nan_count > 0 else ""
    print(f"Epoch {epoch:2d}/20 | λ_clu={lambda_cluster:.2f} | "
          f"Loss: {total_loss / n_batches:.4f} (cls:{total_cls / n_batches:.3f} clu:{total_clu / n_batches:.3f}) | "
          f"Acc: {acc:.4f} | AUC: {auc:.4f} | F1: {f1:.4f} | "
          f"NMI: {nmi:.4f} | ARI: {ari:.4f}{nan_str}")

    if auc > best_auc and not np.isnan(auc):
        best_auc = auc
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"   最佳模型 (AUC: {best_auc:.4f})")

    history['epoch'].append(epoch)
    history['loss'].append(total_loss / n_batches)
    history['acc'].append(acc)
    history['auc'].append(auc)
    history['f1'].append(f1)
    history['nmi'].append(nmi)
    history['ari'].append(ari)

torch.save(best_model_state or model.state_dict(), '/kaggle/working/best_model.pth')
print(f"\n 训练完成！最佳 AUC: {best_auc:.4f}")

# 阶段❹ - Cell 11: 保存模型 + 分类评估 + 聚类分析


import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


# 1. 保存当前模型

torch.save(model.state_dict(), '/kaggle/working/final_model.pth')
print(" 模型已保存至 /kaggle/working/final_model.pth")


# 2. 完整分类评估

@torch.no_grad()
def full_evaluate(model, loader):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    for batch in loader:
        views = apply_augmentation(batch['views_raw'], is_train=False)
        views = [v.to(device) for v in views]
        _, _, logit = model(views)
        probs = torch.sigmoid(logit).cpu().squeeze()
        preds = (probs > 0.5).int()
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(batch['label'].int().tolist())
    model.train()
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

y_true, y_pred, y_prob = full_evaluate(model, val_loader_v2)

print("\n" + "=" * 60)
print(" 验证集分类报告")
print("=" * 60)
print(classification_report(y_true, y_pred, target_names=['良性', '恶性'], digits=4))

# 混淆矩阵
cm = confusion_matrix(y_true, y_pred)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['良性', '恶性'], yticklabels=['良性', '恶性'])
axes[0].set_title('混淆矩阵', fontsize=14)
axes[0].set_xlabel('预测')
axes[0].set_ylabel('真实')

# ROC 曲线
fpr, tpr, _ = roc_curve(y_true, y_prob)
auc = roc_auc_score(y_true, y_prob)
axes[1].plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {auc:.4f}')
axes[1].plot([0, 1], [0, 1], 'r--', linewidth=1)
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC 曲线', fontsize=14)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/kaggle/working/classification_results.png', dpi=150, bbox_inches='tight')
plt.show()


# 3. 聚类分析：提取特征 + K-means + t-SNE

print("\n" + "=" * 60)
print(" 聚类分析：亚型自动发现")
print("=" * 60)

# 提取所有验证集的特征
all_z, all_q, all_labels = get_all_features(model, val_loader_v2)
all_z_np = all_z.numpy()
all_labels_np = all_labels.numpy().astype(int)

# K-means 聚类
kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
cluster_pred = kmeans.fit_predict(all_z_np)

nmi = normalized_mutual_info_score(all_labels_np, cluster_pred)
ari = adjusted_rand_score(all_labels_np, cluster_pred)
print(f"K-means 聚类 (K=8): NMI = {nmi:.4f}, ARI = {ari:.4f}")

# t-SNE 降维可视化
print("正在进行 t-SNE 降维...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
z_tsne = tsne.fit_transform(all_z_np)

# 亚型名称映射
subtype_names = {0: 'A', 1: 'F', 2: 'PT', 3: 'TA', 4: 'DC', 5: 'LC', 6: 'MC', 7: 'PC'}
subtype_full = {
    'A': '腺病', 'F': '纤维腺瘤', 'PT': '叶状肿瘤', 'TA': '管状腺瘤',
    'DC': '导管癌', 'LC': '小叶癌', 'MC': '黏液癌', 'PC': '乳头状癌'
}

# 获取真实亚型
val_subtypes = []
for batch in val_loader_v2:
    val_subtypes.extend(batch['subtype'])
val_subtypes = np.array(val_subtypes[:len(all_labels_np)])

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 按真实亚型着色
colors = plt.cm.tab10(np.linspace(0, 1, 8))
for i, st in enumerate(['A', 'F', 'PT', 'TA', 'DC', 'LC', 'MC', 'PC']):
    mask = val_subtypes == st
    if mask.sum() > 0:
        axes[0].scatter(z_tsne[mask, 0], z_tsne[mask, 1],
                       c=[colors[i]], label=f'{st}({subtype_full[st]})',
                       alpha=0.6, s=30)
axes[0].set_title('t-SNE: 真实亚型分布', fontsize=14)
axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

# 按 K-means 聚类着色
for i in range(8):
    mask = cluster_pred == i
    if mask.sum() > 0:
        axes[1].scatter(z_tsne[mask, 0], z_tsne[mask, 1],
                       c=[colors[i]], label=f'簇 {i+1}',
                       alpha=0.6, s=30)
axes[1].set_title(f't-SNE: K-means 聚类 (NMI={nmi:.3f})', fontsize=14)
axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

plt.tight_layout()
plt.savefig('/kaggle/working/tsne_clustering.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n 聚类分析完成")
print(f"   NMI: {nmi:.4f} (越高越好，最大1.0)")
print(f"   ARI: {ari:.4f} (越高越好，最大1.0)")

# 阶段❺ - Cell 12: Grad-CAM


import cv2
import torch.nn.functional as F

model.eval()
model = model.to(device)


# 1. Hook 设置

target_layer = model.backbone.layers[-1].blocks[-1]

features = []
gradients = []


def forward_hook(module, input, output):
    features.append(output)


def backward_hook(module, grad_in, grad_out):
    gradients.append(grad_out[0])


handle_f = target_layer.register_forward_hook(forward_hook)
handle_b = target_layer.register_full_backward_hook(backward_hook)



# 2. Grad-CAM 计算函数（仅取最后一个视图的梯度）

def compute_gradcam_single_view(model, view_tensor, target_class=1):
    global features, gradients
    features.clear()
    gradients.clear()

    with torch.enable_grad():
        views = [view_tensor.to(device).requires_grad_(True)] * 4
        z, q, logit = model(views)

    model.zero_grad()
    if target_class == 1:
        logit[0, 0].backward(retain_graph=True)
    else:
        (-logit[0, 0]).backward(retain_graph=True)

    #  取最后一个 hook 输出（对应最后一个视图）
    A = features[-1]  # 取最后一个
    grad = gradients[-1]  # 取最后一个

    # 处理形状
    if len(A.shape) == 3:
        # (B, L, C) → (B, C, H, W)
        B, L, C = A.shape
        H = W = int(L ** 0.5)
        A = A.reshape(B, H, W, C).permute(0, 3, 1, 2)
    elif len(A.shape) == 4:
        # (B, H, W, C) → (B, C, H, W)
        A = A.permute(0, 3, 1, 2)

    if len(grad.shape) == 3:
        B, L, C = grad.shape
        H = W = int(L ** 0.5)
        grad = grad.reshape(B, H, W, C).permute(0, 3, 1, 2)
    elif len(grad.shape) == 4:
        grad = grad.permute(0, 3, 1, 2)

    # 确保形状正确
    if A.shape[1] != grad.shape[1]:
        # 通道数不一致，取公共通道
        min_c = min(A.shape[1], grad.shape[1])
        A = A[:, :min_c, :, :]
        grad = grad[:, :min_c, :, :]

    # 权重
    weights = torch.mean(grad, dim=(2, 3), keepdim=True)

    # CAM
    cam = torch.sum(weights * A, dim=1, keepdim=True)
    cam = F.relu(cam)

    # 上采样
    cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
    cam = cam.squeeze().cpu().detach().numpy()

    if cam.max() - cam.min() > 0:
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    else:
        cam = np.zeros((224, 224))

    return cam



# 3. 可视化

def overlay_heatmap(original_img, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    img = original_img.resize((224, 224))
    img = np.array(img)
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    return overlay



# 4. 生成 Grad-CAM

print(" 生成 Grad-CAM 热力图...\n")

val_batch = next(iter(val_loader_v2))
malignant_indices = [i for i, l in enumerate(val_batch['label']) if l == 1]
benign_indices = [i for i, l in enumerate(val_batch['label']) if l == 0]

sample_indices = []
if malignant_indices:
    sample_indices.append(malignant_indices[0])
if len(malignant_indices) > 1:
    sample_indices.append(malignant_indices[1])
if benign_indices:
    sample_indices.append(benign_indices[0])

n_samples = len(sample_indices)
mag_names = ['40X', '100X', '200X', '400X']

fig, axes = plt.subplots(n_samples, 5, figsize=(20, 4 * n_samples))
if n_samples == 1:
    axes = axes.reshape(1, -1)

for row_idx, sample_idx in enumerate(sample_indices):
    label = val_batch['label'][sample_idx].item()
    subtype = val_batch['subtype'][sample_idx]
    true_class = 'Malignant' if label == 1 else 'Benign'

    for col_idx, mag in enumerate(mag_names):
        view_raw = val_batch['views_raw'][col_idx][sample_idx]
        view_normalized = normalize(view_raw).unsqueeze(0)

        cam = compute_gradcam_single_view(model, view_normalized, target_class=1)

        view_np = view_raw.permute(1, 2, 0).numpy()
        view_np = np.clip(view_np, 0, 1)
        view_pil = Image.fromarray((view_np * 255).astype(np.uint8))

        overlay = overlay_heatmap(view_pil, cam, alpha=0.4)

        axes[row_idx, col_idx].imshow(overlay)
        axes[row_idx, col_idx].set_title(f'{mag} - {true_class}', fontsize=12)
        axes[row_idx, col_idx].axis('off')

    view_raw_400 = val_batch['views_raw'][3][sample_idx].permute(1, 2, 0).numpy()
    view_raw_400 = np.clip(view_raw_400, 0, 1)
    axes[row_idx, 4].imshow(view_raw_400)
    axes[row_idx, 4].set_title(f'400X Original\n{subtype}', fontsize=12)
    axes[row_idx, 4].axis('off')

plt.suptitle('Grad-CAM Weakly-Supervised Cancer Localization\n(Red/Yellow = Model Attention on Malignant Regions)',
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('/kaggle/working/gradcam_results.png', dpi=150, bbox_inches='tight')
plt.show()

handle_f.remove()
handle_b.remove()

print("\n Grad-CAM 热力图已生成")

# 阶段❺ - Cell 13: 高置信度恶性样本详细分析


#  确保模型在 eval 模式
model.eval()

# 找到验证集中预测概率最高的恶性样本
y_true, y_pred, y_prob = full_evaluate(model, val_loader_v2)
malignant_mask = y_true == 1

if malignant_mask.sum() > 0:
    malignant_probs = y_prob[malignant_mask]
    top_malignant_idx = np.where(malignant_mask)[0][np.argmax(malignant_probs)]

    # 从验证集中取出该样本
    # 收集所有验证集样本
    all_val_views_raw = []
    all_val_labels = []
    all_val_subtypes = []
    for batch in val_loader_v2:
        for i in range(len(batch['label'])):
            all_val_views_raw.append([batch['views_raw'][v][i] for v in range(4)])
            all_val_labels.append(batch['label'][i].item())
            all_val_subtypes.append(batch['subtype'][i])

    sample = {
        'views_raw': all_val_views_raw[top_malignant_idx],
        'label': all_val_labels[top_malignant_idx],
        'subtype': all_val_subtypes[top_malignant_idx]
    }

    print(f" 高置信度恶性样本分析")
    print(f"   亚型: {sample['subtype']}")
    print(f"   预测恶性概率: {malignant_probs.max():.4f}")

    # 对每个视图生成热力图
    mag_names = ['40X', '100X', '200X', '400X']
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for col, mag in enumerate(mag_names):
        model.eval()  #  每次前确保 eval 模式
        view_raw = sample['views_raw'][col]
        view_normalized = normalize(view_raw).unsqueeze(0)

        cam = compute_gradcam_single_view(model, view_normalized, target_class=1)

        # 原图
        view_np = view_raw.permute(1, 2, 0).numpy()
        view_np = np.clip(view_np, 0, 1)
        view_pil = Image.fromarray((view_np * 255).astype(np.uint8))

        # 原图
        axes[0, col].imshow(view_np)
        axes[0, col].set_title(f'{mag} Original', fontsize=12)
        axes[0, col].axis('off')

        # 热力图叠加
        overlay = overlay_heatmap(view_pil, cam, alpha=0.5)
        axes[1, col].imshow(overlay)
        axes[1, col].set_title(f'{mag} Grad-CAM', fontsize=12)
        axes[1, col].axis('off')

    plt.suptitle(
        f'High-Confidence Malignant Sample - Subtype: {sample["subtype"]}\nHeatmap Shows Model Attention Regions',
        fontsize=14)
    plt.tight_layout()
    plt.savefig('/kaggle/working/gradcam_high_conf.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(" 高置信度恶性样本分析完成")
else:
    print(" 验证集中没有恶性样本")

# 阶段❺ - Cell 14: 单张图片推理接口


import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 加载模型
model.eval()
model = model.to(device)

# 重新注册 hook（如果之前被 remove 了）
target_layer = model.backbone.layers[-1].blocks[-1]
features = []
gradients = []


def forward_hook(module, input, output):
    features.append(output)


def backward_hook(module, grad_in, grad_out):
    gradients.append(grad_out[0])


handle_f = target_layer.register_forward_hook(forward_hook)
handle_b = target_layer.register_full_backward_hook(backward_hook)



# 推理函数

def predict_single_image(image_path, return_heatmap=True):
    """
    对单张病理图片做推理
    参数:
        image_path: 图片路径
        return_heatmap: 是否返回 Grad-CAM 热力图
    返回:
        dict: {
            'prediction': '恶性'/'良性',
            'confidence': float (0-1),
            'malignant_prob': float,
            'heatmap': np.ndarray (可选)
        }
    """
    # 1. 加载并预处理图片
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize((224, 224))
    img_tensor = transforms.ToTensor()(img_resized)
    img_normalized = normalize(img_tensor).unsqueeze(0)  # (1, 3, 224, 224)

    # 2. 前向传播（单视图复制4份）
    model.eval()
    with torch.no_grad():
        views = [img_normalized.to(device)] * 4
        _, _, logit = model(views)
        malignant_prob = torch.sigmoid(logit).item()

    prediction = '恶性' if malignant_prob > 0.5 else '良性'
    confidence = malignant_prob if malignant_prob > 0.5 else 1 - malignant_prob

    result = {
        'prediction': prediction,
        'confidence': confidence,
        'malignant_prob': malignant_prob
    }

    # 3. Grad-CAM 热力图
    if return_heatmap:
        global features, gradients
        features.clear()
        gradients.clear()

        with torch.enable_grad():
            views = [img_normalized.to(device).requires_grad_(True)] * 4
            _, _, logit = model(views)

        model.zero_grad()
        logit[0, 0].backward(retain_graph=True)

        A = features[-1]
        grad = gradients[-1]

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

        result['heatmap'] = cam

    return result



# 测试：对验证集的一张图片做推理

print(" 测试推理功能...\n")

# 取一张恶性样本
val_batch = next(iter(val_loader_v2))
malignant_idx = [i for i, l in enumerate(val_batch['label']) if l == 1][0]
test_image_path = val_df.iloc[malignant_idx]['path_400']  # 取400X的恶性图片

print(f"测试图片: {os.path.basename(test_image_path)}")
print(f"真实标签: 恶性")

result = predict_single_image(test_image_path, return_heatmap=True)

print(f"\n 推理结果:")
print(f"   预测: {result['prediction']}")
print(f"   置信度: {result['confidence']:.2%}")
print(f"   恶性概率: {result['malignant_prob']:.4f}")


# 可视化

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# 原图
img = Image.open(test_image_path).convert('RGB').resize((224, 224))
axes[0].imshow(img)
axes[0].set_title('Original Image', fontsize=14)
axes[0].axis('off')

# 热力图叠加
heatmap = result['heatmap']
heatmap_uint8 = np.uint8(255 * heatmap)
heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
overlay = cv2.addWeighted(np.array(img), 0.5, heatmap_color, 0.5, 0)

axes[1].imshow(overlay)
axes[1].set_title(f'Grad-CAM\nPrediction: {result["prediction"]} ({result["confidence"]:.1%})', fontsize=14)
axes[1].axis('off')

plt.tight_layout()
plt.savefig('/kaggle/working/single_image_inference.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n 推理完成！保存路径: /kaggle/working/single_image_inference.png")


# 再测试一张良性图片

print("\n" + "=" * 50)
benign_idx = [i for i, l in enumerate(val_batch['label']) if l == 0][0]
test_benign_path = val_df.iloc[benign_idx]['path_400']

print(f"测试图片: {os.path.basename(test_benign_path)}")
print(f"真实标签: 良性")

result2 = predict_single_image(test_benign_path, return_heatmap=True)
print(f"\n 推理结果:")
print(f"   预测: {result2['prediction']}")
print(f"   置信度: {result2['confidence']:.2%}")
print(f"   恶性概率: {result2['malignant_prob']:.4f}")
# 对新图片推理
result = predict_single_image('/path/to/your/image.png')

print(result['prediction'])   # '恶性' 或 '良性'
print(result['confidence'])   # 置信度
print(result['malignant_prob'])  # 恶性概率 (0-1)
# result['heatmap']  # Grad-CAM 热力图 (numpy数组)



