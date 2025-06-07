import copy
import os
import random
import shutil
import zipfile
from math import atan2, cos, sin, sqrt, pi, log

import PIL
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from numpy import linalg as LA
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from tqdm import tqdm

# The dataset if downloaded from the following url:
#https://www.kaggle.com/datasets/shuvoalok/cityscapes

import cv2
import os
# #img=cv2.imread(os.path.join(r"/mnt/d/PycharmProjects/Natural_Language_Processing_with_Transformers_Book/data/cityspaces/train","img","train4.png"))
# img=Image.open(os.path.join(r"/mnt/d/PycharmProjects/Natural_Language_Processing_with_Transformers_Book/data/cityspaces/train","img","train4.png"))
# print(img.size)
# #img_label=cv2.imread(os.path.join(r"/mnt/d/PycharmProjects/Natural_Language_Processing_with_Transformers_Book/data/cityspaces/train","label","train4.png"))
# img_label=Image.open(os.path.join(r"/mnt/d/PycharmProjects/Natural_Language_Processing_with_Transformers_Book/data/cityspaces/train","label","train4.png"))
# print(img_label.size)

# plt.imshow(img_label)
# plt.show()

import numpy as np
# def transform_road_surface(label_image:PIL.Image.Image)->PIL.Image.Image:
# # Convert your image to float for distance computation
#     #label_image=np.array(label_image)
#     #label_image_rgb = cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB)
#     label_image = np.array(label_image) #.astype(np.float32)
#     # Define the road color in RGB
#     target_color = np.array([128, 64, 128], dtype=np.float32)
#     # Compute per-pixel color distance (Euclidean)
#     distance = np.linalg.norm(label_image - target_color, axis=-1)
#     # Define threshold (tune as needed; try 10–20 for JPEGs)
#     threshold = 10.0
#     # Create mask: True where pixel color is close to road color
#     mask = distance < threshold  # shape: (H, W)
#     # Create output image with same shape, initialized to black
#     filtered_img = np.zeros_like(label_image)
#     # Copy only the road-like pixels
#     filtered_img[mask] = label_image[mask]
#     return Image.fromarray(filtered_img.astype(np.uint8))
def transform_car_surface(label_image: PIL.Image.Image) -> PIL.Image.Image:
    # 转换为 RGB，确保是三通道
    label_image = label_image.convert("RGB")
    label_array = np.array(label_image)

    # Cityscapes 中 car 类的颜色是 (0, 0, 142)
    car_color = np.array([0, 0, 142])

    # 创建一个 mask：像素颜色接近 car_color 的地方置为 1
    distance = np.linalg.norm(label_array - car_color, axis=-1)
    mask = (distance < 10).astype(np.uint8)  # 距离阈值越小匹配越严格

    # 转换为 0 和 255 的图像
    return Image.fromarray(mask * 255)  # 单通道

# test_img=transform_road_surface(img_label).convert('L')
# trans=transforms.Compose([
#             #transforms.Resize((512, 512)),
#             transforms.Resize((96,256)),
#             transforms.ToTensor()])
# img1=trans(test_img)
# plt.imshow(test_img,cmap='gray')
# plt.show()



class CitySpacesDataset(Dataset):
    def __init__(self, image_root, label_root, limit=None):
        self.images = []
        self.masks = []
        self.limit = limit

        for city in os.listdir(image_root):
            city_img_path = os.path.join(image_root, city)
            city_label_path = os.path.join(label_root, city)

            for file in os.listdir(city_img_path):
                if file.endswith("_leftImg8bit.png"):
                    img_path = os.path.join(city_img_path, file)
                    
                    label_file = file.replace("_leftImg8bit.png", "_gtFine_color.png")
                    label_path = os.path.join(city_label_path, label_file)

                    if os.path.exists(label_path):
                        self.images.append(img_path)
                        self.masks.append(label_path)

        if self.limit is not None:
            self.images = self.images[:self.limit]
            self.masks = self.masks[:self.limit]

        self.transform = transforms.Compose([
            transforms.Resize((96, 256)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index])
        mask = transform_car_surface(mask).convert("L")
        return self.transform(img), self.transform(mask)

    def __len__(self):
        return len(self.images)

train_dataset = CitySpacesDataset(
    image_root="leftImg8bit/train",
    label_root="gtFine/train")
test_dataset = CitySpacesDataset(
    image_root="leftImg8bit/test",
    label_root="gtFine/test")
# TRAIN_DIR=r"/mnt/d/PycharmProjects/Natural_Language_Processing_with_Transformers_Book/data/cityspaces/train"
# TEST_DIR=r"/mnt/d/PycharmProjects/Natural_Language_Processing_with_Transformers_Book/data/cityspaces/val"
# train_dataset=CitySpacesDataset(TRAIN_DIR)
# test_dataset = CitySpacesDataset(TEST_DIR)
# print(train_dataset[0][0].shape)
# print(train_dataset[0][1].shape)
num_workers=0
device = "cuda" if torch.cuda.is_available() else "cpu"
if device=="cuda":
    num_workers=torch.cuda.device_count()*4
LEARNING_RATE=3e-4
BATCH_SIZE=8
generator=torch.Generator().manual_seed(42)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers,generator=generator)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers,generator=generator)

class DoubleConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv_op=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv_op(x)

class DownSample(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv=DoubleConv(in_channels,out_channels)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
    def forward(self,x):
        down=self.conv(x)
        p=self.pool(down)
        return down,p

class UpSample(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.up=nn.ConvTranspose2d(in_channels,in_channels//2,kernel_size=2,stride=2)
        self.conv=DoubleConv(in_channels,out_channels)
    def forward(self,x1,x2):
        x1=self.up(x1)
        x=torch.cat([x1,x2],1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self,in_channels,num_classes):
        super().__init__()
        self.down_conv1=DownSample(in_channels,64)
        self.down_conv2=DownSample(64,128)
        self.down_conv3=DownSample(128,256)
        self.down_conv4=DownSample(256,512)

        self.bottle_neck=DoubleConv(512,1024)

        self.up_conv1=UpSample(1024,512)
        self.up_conv2=UpSample(512,256)
        self.up_conv3=UpSample(256,128)
        self.up_conv4=UpSample(128,64)

        self.out=nn.Conv2d(64,num_classes,kernel_size=1)
    def forward(self,x):
        down1,p1=self.down_conv1(x)
        down2,p2=self.down_conv2(p1)
        down3,p3=self.down_conv3(p2)
        down4,p4=self.down_conv4(p3)
        b=self.bottle_neck(p4)
        up1=self.up_conv1(b,down4)
        up2=self.up_conv2(up1,down3)
        up3=self.up_conv3(up2,down2)
        up4=self.up_conv4(up3,down1)
        out=self.out(up4)
        return out
model = UNet(in_channels=3, num_classes=1).to(device)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCEWithLogitsLoss()

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

torch.save(model.state_dict(), 'dcs_rover_checkpoint.pth')
##############################################################################################################
inference_transform = transforms.Compose([
    transforms.Resize((96, 256)),
    transforms.ToTensor()
])
model.eval()
#img_path = r"/mnt/d/PycharmProjects/Natural_Language_Processing_with_Transformers_Book/data/cityspaces/val/img/val66.png"  # replace with your image path
img_path=r"R.jpg"
img = Image.open(img_path).convert("RGB")
input_tensor = inference_transform(img).unsqueeze(0).to(device)  # add batch dimension
with torch.no_grad():
    output = model(input_tensor)
    output = torch.sigmoid(output)  # Apply sigmoid to get probabilities
    output = output.squeeze().cpu().numpy()

plt.figure(figsize=(5, 5))
# Original image
plt.subplot(2, 1, 1)
plt.title("Original Image")
plt.imshow(img)
plt.axis('off')
# Predicted mask
plt.subplot(2, 1, 2)
plt.title("Predicted Mask")
plt.imshow(output, cmap='gray')
plt.axis('off')
plt.show()
output_path = "/root/workspace"
os.makedirs(output_path, exist_ok=True)

# 保存原始图像
original_img_path = os.path.join(output_path, "original_image.png")
plt.imsave(original_img_path, img)

# 保存预测掩膜
predicted_mask_path = os.path.join(output_path, "predicted_mask.png")
plt.imsave(predicted_mask_path, output, cmap='gray')

print(f"Results saved to {output_path}")
###################################################################################################
# output = model(input_tensor)
# output=torch.sigmoid(output)
# mask = output.squeeze().detach().cpu().numpy()
# binary_mask = (mask > 0.3).astype(np.uint8)
































# # 2. Make a copy of the original image (so we don't modify it directly)
# overlay_img = mask.copy()

# # 3. Draw contours in green (you can change the color and thickness)
# contour_img=cv2.drawContours(np.array(overlay_img), contours, -1, (0, 255, 0), 2)  # -1 means draw all contours

# # 4. (Optional) Display using matplotlib
# import matplotlib.pyplot as plt

# # If original_img is BGR (as from OpenCV), convert to RGB for correct display
# overlay_rgb = cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB)

# plt.imshow(overlay_rgb)
# plt.title("Lane Contours Overlayed")
# plt.axis('off')
# plt.show()





# import cv2

# contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# if len(contours) == 0:
#     print("No road detected")
#     direction_vector = None
# else:
#     largest_contour = max(contours, key=cv2.contourArea)

# [vx, vy, x0, y0] = cv2.fitLine(largest_contour, cv2.DIST_L2, 0, 0.01, 0.01)
# # vx, vy is the direction vector of the line (unit vector)

# import matplotlib.pyplot as plt

# img_color = np.array(img)  # original image
# pt1 = (int(x0 - vx * 50), int(y0 - vy * 50))
# pt2 = (int(x0 + vx * 50), int(y0 + vy * 50))

# img_with_arrow = img_color.copy()
# cv2.arrowedLine(img_with_arrow, pt1, pt2, color=(255, 0, 0), thickness=2)

# plt.imshow(img_with_arrow)
# plt.title("Estimated Direction of Travel")
# plt.axis('off')
# plt.show()





