import os
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor, AdamW

# -----------------------------
# 1. 数据集定义（只保留车辆类）
# -----------------------------
class CityscapesCarDataset(Dataset):
    def __init__(self, image_root, label_root, transform_img=None, transform_mask=None):
        self.image_paths = []
        self.label_paths = []
        self.transform_img = transform_img
        self.transform_mask = transform_mask

        for city in os.listdir(image_root):
            city_img_path = os.path.join(image_root, city)
            city_label_path = os.path.join(label_root, city)

            for file_name in os.listdir(city_img_path):
                if file_name.endswith('_leftImg8bit.png'):
                    img_path = os.path.join(city_img_path, file_name)
                    label_file = file_name.replace('_leftImg8bit.png', '_gtFine_labelIds.png')
                    label_path = os.path.join(city_label_path, label_file)

                    if os.path.exists(label_path):
                        self.image_paths.append(img_path)
                        self.label_paths.append(label_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = Image.open(self.label_paths[idx])

        label = np.array(label)
        mask = np.isin(label, [26, 27, 28, 31]).astype(np.uint8)  # 包括车辆类
        mask = Image.fromarray(mask * 255)

        if self.transform_img:
            image = self.transform_img(image)
        if self.transform_mask:
            mask = self.transform_mask(mask)

        return image, mask

# -----------------------------
# 2. 数据预处理
# -----------------------------
image_transform = transforms.Compose([
    transforms.Resize((256, 512)),  # 可改为 (96, 256) 与 UNet 对齐
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

mask_transform = transforms.Compose([
    transforms.Resize((256, 512), interpolation=Image.NEAREST),
    transforms.ToTensor(),  # 输出为 [1, H, W]，0 或 1
])

# -----------------------------
# 3. Dice Loss 函数
# -----------------------------
def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(1)
    dice = (2. * intersection + smooth) / (pred_flat.sum(1) + target_flat.sum(1) + smooth)
    return 1 - dice.mean()

# -----------------------------
# 4. Dice Score
# -----------------------------
def dice_score(pred, target, threshold=0.5):
    pred = torch.sigmoid(pred)
    pred_bin = (pred > threshold).float()
    intersection = (pred_bin * target).sum(dim=(1, 2, 3))
    union = pred_bin.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * intersection) / (union + 1e-8)
    return dice.mean().item()

# -----------------------------
# 5. 训练函数
# -----------------------------
def train_segformer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据加载
    dataset = CityscapesCarDataset(
        image_root='leftImg8bit/train',
        label_root='gtFine/train',
        transform_img=image_transform,
        transform_mask=mask_transform
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    # 模型

    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=1,
        ignore_mismatched_sizes=True
    ).to(device)
    model.save_pretrained("./local_segformer/")

    optimizer = AdamW(model.parameters(), lr=3e-5)
    bce_loss = nn.BCEWithLogitsLoss()

    # 训练
    model.train()
    num_epochs = 60

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_dice = 0.0

        for images, masks in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, masks = images.to(device), masks.to(device)

            outputs = model(pixel_values=images).logits  # [B, 1, H, W]
            outputs = F.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)  # 上采样回 mask 尺寸
            loss_bce = bce_loss(outputs, masks)
            loss_dice = dice_loss(outputs, masks)
            loss = loss_bce + loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_dice += dice_score(outputs.detach(), masks)

        avg_loss = total_loss / len(dataloader)
        avg_dice = total_dice / len(dataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Dice Score: {avg_dice:.4f}")

    # 保存模型
    torch.save(model.state_dict(), "segformer_car.pth")
    print("✅ 模型已保存为 segformer_car.pth")

# -----------------------------
# 启动训练
# -----------------------------
if __name__ == "__main__":
    train_segformer()
