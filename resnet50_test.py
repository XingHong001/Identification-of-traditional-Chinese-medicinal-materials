import os
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# 设置随机种子确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 数据预处理设置
# 测试集不使用数据增强，仅进行归一化
test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 自定义数据集类，适应直接存放图片的目录结构
class CustomImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        # 只保留图片文件，排除目录和其他文件
        self.image_files = [f for f in os.listdir(data_dir) 
                          if os.path.isfile(os.path.join(data_dir, f)) and 
                          f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, 0
        except Exception as e:
            print(f"无法加载图像 {img_path}: {str(e)}")
            # 返回一个空图像作为占位符
            return torch.zeros(3, 224, 224), 0

# Grad-CAM实现
class GradCAM:
    """
    Grad-CAM实现类，用于生成类别激活热力图
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradient = None

        # 注册钩子
        self.hook_handles = []

        # 保存特征图的正向钩子
        def forward_hook(module, input, output):
            self.feature_maps = output.detach()

        # 保存梯度的反向钩子
        def backward_hook(module, grad_in, grad_out):
            self.gradient = grad_out[0].detach()

        # 获取目标层
        target_found = False
        for name, module in self.model.named_modules():
            if name == target_layer:
                self.hook_handles.append(module.register_forward_hook(forward_hook))
                self.hook_handles.append(module.register_backward_hook(backward_hook))
                target_found = True
                break

        if not target_found:
            raise ValueError(f"目标层 {target_layer} 未在模型中找到")

    def __call__(self, x, class_idx=None):
        # 确保模型处于评估模式
        self.model.eval()

        # 前向传播
        x = x.requires_grad_()
        output = self.model(x)

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1)

        # 目标是最大化该类别的分数
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1

        # 反向传播
        self.model.zero_grad()
        output.backward(gradient=one_hot, retain_graph=True)

        # 计算权重 (全局平均池化梯度)
        weights = torch.mean(self.gradient, dim=(2, 3), keepdim=True)

        # 加权组合特征图
        cam = torch.sum(weights * self.feature_maps, dim=1).squeeze()

        # ReLU激活，因为我们只关心正面影响
        cam = torch.clamp(cam, min=0)

        # 归一化
        if torch.max(cam) > 0:
            cam = cam / torch.max(cam)

        # 调整为与输入图像相同的尺寸
        cam = torch.nn.functional.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(x.size(2), x.size(3)),
            mode='bilinear',
            align_corners=False
        ).squeeze()

        return cam.detach().cpu().numpy()

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

# 可视化Grad-CAM结果并保存
def visualize_grad_cam(model, img_path, class_names, save_dir, target_layer='layer4.2.conv3'):
    """使用Grad-CAM可视化模型关注区域并保存结果"""
    # 加载并预处理图像
    img = Image.open(img_path).convert('RGB')
    img_tensor = test_transforms(img).unsqueeze(0).to(device)

    # 创建Grad-CAM对象
    grad_cam = GradCAM(model, target_layer)

    # 获取预测结果
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        _, preds = torch.max(outputs, 1)
        pred_class = class_names[preds.item()]
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf = probs[0, preds.item()].item() * 100

    # 生成Grad-CAM热力图
    cam = grad_cam(img_tensor, preds.item())
    grad_cam.remove_hooks()

    # 准备显示
    plt.figure(figsize=(12, 4))

    # 显示原始图像
    plt.subplot(131)
    plt.imshow(img)
    plt.title('原始图像')
    plt.axis('off')

    # 显示热力图
    plt.subplot(132)
    plt.imshow(cam, cmap='jet')
    plt.title('Grad-CAM热力图')
    plt.axis('off')

    # 显示叠加结果
    plt.subplot(133)
    img_np = np.array(img.resize((224, 224)))
    heatmap = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    superimposed_img = heatmap * 0.4 + img_np
    superimposed_img = np.uint8(superimposed_img)
    plt.imshow(superimposed_img)
    plt.title(f'预测: {pred_class} ({conf:.1f}%)')
    plt.axis('off')

    plt.tight_layout()

    # 保存图像
    img_name = os.path.basename(img_path)
    save_path = os.path.join(save_dir, img_name)
    plt.savefig(save_path)
    plt.close()

# 加载数据集
def load_data(data_dir):
    """加载图像数据集"""
    dataset = CustomImageDataset(data_dir, test_transforms)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    dataset_size = len(dataset)
    class_names = ['aiye','ajiao','baibiandou','baibu'] 

    return dataloader, dataset_size, class_names

# 在测试集上评估模型
def evaluate_model(model, dataloader, dataset_size):
    model.eval()
    running_corrects = 0
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == torch.tensor([0] * len(preds)).to(device))

    test_acc = running_corrects.double() / dataset_size
    print(f'总体 Test Acc: {test_acc:.4f}')

# 测试单张图像
def test_single_image(model, img_path, class_names, save_dir):
    model.eval()
    img = Image.open(img_path).convert('RGB')
    img_tensor = test_transforms(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, preds = torch.max(outputs, 1)
        pred_class = class_names[preds.item()]
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf = probs[0, preds.item()].item() * 100

        print(f'预测结果: {pred_class}，置信度: {conf:.2f}%')

    # 可视化Grad-CAM结果并保存
    visualize_grad_cam(model, img_path, class_names, save_dir)

def plot_predictions(model, dataloader, class_names, num_samples=9):
    """可视化模型预测结果"""
    model.eval()
    images, labels, preds = [], [], []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            images.extend(inputs.cpu().numpy())
            labels.extend(targets.cpu().numpy())
            preds.extend(predicted.cpu().numpy())
            
            if len(images) >= num_samples:
                break
    
    plt.figure(figsize=(12, 12))
    for i in range(num_samples):
        plt.subplot(3, 3, i+1)
        img = images[i].transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        plt.imshow(img)
        plt.title(f'真实: {class_names[labels[i]]}\n预测: {class_names[preds[i]]}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_samples.png')
    plt.show()


def plot_gradient_distribution(model):
    """绘制模型各层的梯度分布"""
    gradients = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients.append(param.grad.norm().item())
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(gradients)), gradients)
    plt.title('模型各层梯度分布')
    plt.xlabel('层索引')
    plt.ylabel('梯度范数')
    plt.savefig('gradient_distribution.png')
    plt.show()


def plot_training_metrics(history):
    """绘制训练和验证的损失率、准确率曲线"""
    plt.figure(figsize=(12, 5))
    
    # 损失率曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.title('训练和验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='训练准确率')
    plt.plot(history['val_acc'], label='验证准确率')
    plt.title('训练和验证准确率')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()

if __name__ == "__main__":
    # 设置数据集路径，需要根据实际情况修改
    data_dir = os.path.abspath("test")  # 使用绝对路径
    
    # 检查测试目录是否存在
    if not os.path.exists(data_dir):
        print(f"错误: 测试目录 {data_dir} 不存在")
        exit(1)
        
    # 检查目录是否可读
    if not os.access(data_dir, os.R_OK):
        print(f"错误: 没有读取 {data_dir} 的权限")
        exit(1)
        
    # 检查目录中是否有图片文件
    if not any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in os.listdir(data_dir)):
        print(f"错误: {data_dir} 中没有找到任何图片文件(.png/.jpg/.jpeg)")
        exit(1)

    # 创建保存热力图的文件夹
    save_dir = 'test_analysis'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载数据
    dataloader, dataset_size, class_names = load_data(data_dir)
    print(f"测试集大小: {dataset_size}")
    print(f"类别名称: {class_names}")

    # 加载预训练的ResNet50模型
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)  # 这里假设只有一个类别（因为数据布局无法区分类别）
    model = model.to(device)

    # 加载保存的模型权重
    model_path = 'resnet50_finetuned.pth'
    try:
        model.load_state_dict(torch.load(model_path))
        print(f"成功加载模型: {model_path}")
    except FileNotFoundError:
        print(f"错误: 模型文件 {model_path} 不存在，请确保已经训练并保存了模型。")
        exit(1)

    # 在测试集上评估模型
    evaluate_model(model, dataloader, dataset_size)

    # 测试所有图像并保存热力图
    test_dataset = dataloader.dataset
    for img_name in test_dataset.image_files:
        img_path = os.path.join(data_dir, img_name)
        print(f"\n测试图像: {img_path}")
        test_single_image(model, img_path, class_names, save_dir)