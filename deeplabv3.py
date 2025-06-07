import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CityscapesCarDataset(Dataset):
    def __init__(self, image_root, label_root, transform=None):
        self.image_paths = []
        self.label_paths = []
        self.transform = transform

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

        # 提取汽车和卡车的标签（ID为26和27）
        label = np.array(label)
        mask = np.isin(label, [26, 27]).astype(np.uint8)
        mask = Image.fromarray(mask * 255)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask
import torchvision.models.segmentation as models
import torch.nn as nn

def get_deeplabv3_model(num_classes):
    model = models.deeplabv3_resnet50(pretrained=True)
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据变换
transform = transforms.Compose([
    transforms.Resize((256, 512)),
    transforms.ToTensor(),
])

# 加载数据集
train_dataset = CityscapesCarDataset(
    image_root='leftImg8bit/train',
    label_root='gtFine/train',
    transform=transform
)
val_dataset = CityscapesCarDataset(
    image_root='leftImg8bit/val',
    label_root='gtFine/val',
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

# 初始化模型
model = get_deeplabv3_model(num_classes=1)
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def dice_coefficient(prediction, target, epsilon=1e-7):
    prediction = torch.sigmoid(prediction)  # <--- Fix here. Otherwise, DICE value would always be 0.
    prediction = (prediction > 0.5).float()
    target = (target > 0.5).float()

    intersection = (prediction * target).sum()
    union = prediction.sum() + target.sum()
    dice = (2. * intersection + epsilon) / (union + epsilon)
    return dice

torch.cuda.empty_cache()

EPOCHS = 5

train_losses = []
train_dcs = []
val_losses = []
val_dcs = []

for epoch in tqdm(range(EPOCHS)):
    model.train()
    train_running_loss = 0
    train_running_dc = 0

    for idx, img_mask in enumerate(tqdm(train_loader, position=0, leave=True)):
        img = img_mask[0].float().to(device) #This is not a singe image but a batch.
        mask = img_mask[1].float().to(device)

        y_pred = model(img)
        optimizer.zero_grad()

        dc = dice_coefficient(y_pred, mask)
        loss = criterion(y_pred, mask)

        train_running_loss += loss.item()
        train_running_dc += dc.item()

        loss.backward()
        optimizer.step()

    train_loss = train_running_loss / (idx + 1)
    train_dc = train_running_dc / (idx + 1)

    train_losses.append(train_loss)
    train_dcs.append(train_dc)


    model.eval()
    val_running_loss = 0
    val_running_dc = 0

    with torch.no_grad():
        for idx, img_mask in enumerate(tqdm(test_loader, position=0, leave=True)):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)

            y_pred = model(img)
            loss = criterion(y_pred, mask)
            dc = dice_coefficient(y_pred, mask)

            val_running_loss += loss.item()
            val_running_dc += dc.item()

        val_loss = val_running_loss / (idx + 1)
        val_dc = val_running_dc / (idx + 1)

    val_losses.append(val_loss)
    val_dcs.append(val_dc)

    print("-" * 30)
    print(f"Training Loss EPOCH {epoch + 1}: {train_loss:.4f}")
    print(f"Training DICE EPOCH {epoch + 1}: {train_dc:.4f}")
    print("\n")
    print(f"Validation Loss EPOCH {epoch + 1}: {val_loss:.4f}")
    print(f"Validation DICE EPOCH {epoch + 1}: {val_dc:.4f}")
    print("-" * 30)

torch.save(model.state_dict(), 'deeplabv3_car.pth')


inference_transform = transforms.Compose([
    transforms.Resize((256,512)),
    transforms.ToTensor()
])

model.eval()
img_path = "R.jpg"
img = Image.open(img_path).convert("RGB")
input_tensor = inference_transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(input_tensor)
    output = torch.sigmoid(output).squeeze().cpu().numpy()
    binary_output = (output > 0.5).astype(np.uint8) * 255

plt.figure(figsize=(5, 5))
plt.subplot(2, 1, 1)
plt.title("Original Image")
plt.imshow(img)
plt.axis('off')

plt.subplot(2, 1, 2)
plt.title("Predicted Mask (Binary)")
plt.imshow(binary_output, cmap='gray')
plt.axis('off')
plt.show()

output_path = "/root/workspace"
os.makedirs(output_path, exist_ok=True)
plt.imsave(os.path.join(output_path, "original_image.png"), img)
plt.imsave(os.path.join(output_path, "predicted_mask.png"), binary_output, cmap='gray')

print(f"Results saved to {output_path}")

