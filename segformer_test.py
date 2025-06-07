import os
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import numpy as np
from transformers import SegformerForSemanticSegmentation
import matplotlib.pyplot as plt

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型结构和训练好的权重
model = SegformerForSemanticSegmentation.from_pretrained(
    "./local_segformer/",
    num_labels=1,
    ignore_mismatched_sizes=True
).to(device)
model.load_state_dict(torch.load("segformer_car.pth", map_location=device))
model.eval()

# 图像预处理（和训练时一致）
transform = transforms.Compose([
    transforms.Resize((256, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 创建保存目录
os.makedirs("results", exist_ok=True)

# 获取测试图片路径
test_dir = "img"
test_images = [f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

# 推理 & 可视化
for idx, filename in enumerate(test_images):
    img_path = os.path.join(test_dir, filename)
    image = Image.open(img_path).convert('RGB')

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(pixel_values=input_tensor).logits
        output = F.interpolate(output, size=(256, 512), mode='bilinear', align_corners=False)
        pred_mask = torch.sigmoid(output)
        pred_mask = (pred_mask.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255

    # 可视化与保存
    # 原图去归一化
    image_np = np.array(image.resize((512, 256)))
    overlay = image_np.copy()
    overlay[pred_mask == 255] = [255, 0, 0]  # 红色叠加
    overlay = (0.5 * image_np + 0.5 * overlay).astype(np.uint8)

    Image.fromarray(image_np).save(f"results/{idx}_orig.png")
    Image.fromarray(pred_mask).save(f"results/{idx}_mask.png")
    Image.fromarray(overlay).save(f"results/{idx}_overlay.png")

    print(f"[✓] 已保存图像：{filename}")

print("全部测试图像处理完毕，结果保存在 ./results/")
