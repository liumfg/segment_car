import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from tqdm import tqdm

def transform_car_surface(label_image: Image.Image) -> Image.Image:
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

def visualize_and_save_comparison(original_img, predicted_mask, output_path, model_name="DeepLabV3"):
    """
    可视化原始图像、预测掩码及它们的叠加效果，并保存为图片
    
    参数:
    original_img: PIL.Image 原始RGB图像
    predicted_mask: numpy.ndarray 预测的掩码（0-1范围）
    output_path: str 保存图像的路径
    model_name: str 模型名称，用于标题显示
    """
    # 将掩码转换为可显示的格式 (0-255)
    mask_display = (predicted_mask * 255).astype(np.uint8)
    
    # 创建颜色掩码（绿色）
    green_mask = np.zeros((mask_display.shape[0], mask_display.shape[1], 3), dtype=np.uint8)
    green_mask[:,:,1] = mask_display  # 绿色通道
    
    # 将PIL图像转换为numpy数组
    img_np = np.array(original_img)
    
    # 调整掩码大小以匹配原始图像
    if img_np.shape[:2] != mask_display.shape[:2]:
        green_mask = cv2.resize(green_mask, (img_np.shape[1], img_np.shape[0]))
    
    # 创建叠加图像 (原始图像 + 绿色掩码)
    overlay = cv2.addWeighted(img_np, 0.7, green_mask, 0.3, 0)
    
    # 创建一个大图像，包含三个子图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 原始图像
    axes[0].imshow(original_img)
    axes[0].set_title('initial image')
    axes[0].axis('off')
    
    # 预测掩码
    axes[1].imshow(mask_display, cmap='gray')
    axes[1].set_title(f'{model_name} mask')
    axes[1].axis('off')
    
    # 叠加效果
    axes[2].imshow(overlay)
    axes[2].set_title('mask+image')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"对比图已保存至: {output_path}")

def main():
    # 配置参数
    MODEL_PATH = "deeplabv3_car.pth"  # 训练好的模型权重路径
    IMAGE_DIR = "img"  # 测试图片所在文件夹
    OUTPUT_DIR = "output"  # 结果保存文件夹
    INPUT_SIZE = (256, 512)  # 模型输入尺寸
    THRESHOLD = 0.5  # 二值化阈值
    
    # 创建输出文件夹
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 设备配置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 加载模型
    # 加载模型
    model = deeplabv3_resnet50(pretrained=False, aux_loss=False)
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)  # 调整输出通道数
    
    # 加载状态字典并过滤不需要的键
    state_dict = torch.load(MODEL_PATH, map_location=device)
    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('aux_classifier.')}
    model.load_state_dict(filtered_state_dict)
    model = model.to(device)
    model.eval()
    # model = deeplabv3_resnet50(pretrained=False)
    # model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)  # 调整输出通道数
    # model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    # model = model.to(device)
    # model.eval()
    
    # 图像转换
    inference_transform = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor()
    ])
    
    # 获取所有测试图片
    image_files = [f for f in os.listdir(IMAGE_DIR) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if not image_files:
        print(f"在 {IMAGE_DIR} 文件夹中未找到图片文件!")
        return
    
    print(f"找到 {len(image_files)} 张图片，开始处理...")
    
    # 处理每张图片
    for image_file in tqdm(image_files):
        image_path = os.path.join(IMAGE_DIR, image_file)
        try:
            # 读取图片
            img = Image.open(image_path).convert("RGB")
            original_size = img.size
            
            # 准备模型输入
            input_tensor = inference_transform(img).unsqueeze(0).to(device)  # 添加批次维度
            
            # 模型推理
            with torch.no_grad():
                output = model(input_tensor)
                output = output['out']  # 提取输出的分割结果
                output = torch.sigmoid(output)  # 应用sigmoid获取概率值
            
            # 处理输出结果
            mask = output.squeeze().cpu().numpy()  # 移除批次和通道维度
            binary_mask = (mask > THRESHOLD).astype(np.uint8)  # 二值化

            
            # 保存原始大小的掩码
            original_mask = Image.fromarray((binary_mask * 255).astype(np.uint8)).resize(original_size)
            
            # 保存对比图
            output_name = os.path.splitext(image_file)[0] + "_comparison.png"
            output_path = os.path.join(OUTPUT_DIR, output_name)
            visualize_and_save_comparison(img, np.array(original_mask)/255.0, output_path)
            
            # 保存仅掩码图像
            mask_name = os.path.splitext(image_file)[0] + "_mask.png"
            mask_path = os.path.join(OUTPUT_DIR, mask_name)
            original_mask.save(mask_path)
            
        except Exception as e:
            print(f"处理图片 {image_file} 时出错: {str(e)}")
    
    print(f"所有图片处理完成，结果保存在 {OUTPUT_DIR} 文件夹中")

if __name__ == "__main__":
    main()