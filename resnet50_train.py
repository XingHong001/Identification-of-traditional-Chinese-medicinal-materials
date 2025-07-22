import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from torchvision.models import ResNet50_Weights

# 设置随机种子确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 数据增强和归一化设置
# 训练集使用数据增强和归一化
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 验证集不使用数据增强，仅进行归一化
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

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
def visualize_grad_cam(model, img_path, class_names, target_layer='layer4.2.conv3', save_path=None):
    """使用Grad-CAM可视化模型关注区域并保存结果"""
    # 加载并预处理图像
    img = Image.open(img_path).convert('RGB')
    img_tensor = val_transforms(img).unsqueeze(0).to(device)
    
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
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

# 加载数据集
def load_data(data_dir):
    """加载图像数据集"""
    image_datasets = {
        'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), train_transforms),
        'val': datasets.ImageFolder(os.path.join(data_dir, 'val'), val_transforms)
    }
    
    # 创建数据加载器
    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4)
        for x in ['train', 'val']
    }
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    
    return dataloaders, dataset_sizes, class_names

# 训练模型函数
def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25):
    """训练模型并返回最优模型"""
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # 每个epoch都有一个训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 训练模式
            else:
                model.eval()   # 评估模式

            running_loss = 0.0
            running_corrects = 0

            # 迭代数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 零参数梯度
                optimizer.zero_grad()

                # 前向传播
                # 只有在训练时才跟踪历史
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 只有在训练阶段才进行反向传播和优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc.cpu().numpy())  # 将 CUDA 张量移到 CPU 并转换为 NumPy 数组
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc.cpu().numpy())  # 将 CUDA 张量移到 CPU 并转换为 NumPy 数组

            # 深拷贝模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # 加载最优模型权重
    model.load_state_dict(best_model_wts)

    # 绘制损失率和准确率图
    plt.figure(figsize=(12, 4))

    plt.subplot(121)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')

    plt.subplot(122)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'loss_accuracy_curve.png'))
    plt.close()

    return model, train_losses, val_losses, train_accs, val_accs

# 可视化模型预测结果并保存拟合图
def visualize_model(model, dataloaders, class_names, num_images=6, save_path=None):
    """可视化模型的预测结果并保存拟合图"""
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(15, 10))
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}, actual: {class_names[labels[j]]}')
                
                # 将归一化的图像还原以便显示
                img = inputs.cpu().data[j].numpy().transpose((1, 2, 0))
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
                img = np.clip(img, 0, 1)
                plt.imshow(img)
                
                if images_so_far == num_images:
                    model.train(mode=was_training)
                    if save_path:
                        plt.savefig(save_path)
                    else:
                        plt.show()
                    plt.close()
                    return
        
        model.train(mode=was_training)

# 过拟合欠拟合分析函数
def analyze_fitting(train_losses, val_losses, train_accs, val_accs, log_dir):
    # 判断过拟合
    if len(val_losses) > 1 and val_losses[-1] > min(val_losses[:-1]) and train_losses[-1] < min(train_losses[:-1]):
        fitting_status = "过拟合"
    # 判断欠拟合
    elif train_accs[-1] < 0.5 and val_accs[-1] < 0.5:
        fitting_status = "欠拟合"
    else:
        fitting_status = "正常拟合"

    # 保存分析结果到文件
    with open(os.path.join(log_dir, 'fitting_analysis.txt'), 'w') as f:
        f.write(f"模型拟合状态: {fitting_status}\n")
        f.write(f"训练集最后一轮损失: {train_losses[-1]:.4f}\n")
        f.write(f"验证集最后一轮损失: {val_losses[-1]:.4f}\n")
        f.write(f"训练集最后一轮准确率: {train_accs[-1]:.4f}\n")
        f.write(f"验证集最后一轮准确率: {val_accs[-1]:.4f}\n")

    print(f"模型拟合状态: {fitting_status}")

# 主函数
if __name__ == "__main__":
    # 设置数据集路径，需要根据实际情况修改
    data_dir = "resnet_data"
    
    # 创建log文件夹
    log_dir = 'log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 加载数据
    dataloaders, dataset_sizes, class_names = load_data(data_dir)
    num_classes = len(class_names)
    print(f"训练集大小: {dataset_sizes['train']}, 验证集大小: {dataset_sizes['val']}")
    print(f"类别数量: {num_classes}, 类别名称: {class_names}")
    
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载预训练的ResNet50模型
    model_ft = models.resnet50(pretrained=True)
    
    # 冻结大部分预训练层
    for param in list(model_ft.parameters())[:-20]:  # 解冻最后20层
        param.requires_grad = False
    
    # 修改最后一层以适应我们的分类任务
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    
    # 将模型移到GPU
    model_ft = model_ft.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)  # 调整学习率和动量 by 0.1
    
    # 学习率调度器
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)  # 每7个epoch衰减学习率 by 0.1 
    
    # 训练模型
    model_ft, train_losses, val_losses, train_accs, val_accs = train_model(
        model_ft, 
        criterion, 
        optimizer_ft, 
        exp_lr_scheduler, 
        dataloaders, 
        dataset_sizes,
        num_epochs=49
    )
    
    # 可视化一些预测结果并保存拟合图
    visualize_model(model_ft, dataloaders, class_names, save_path=os.path.join(log_dir, 'fitting_plot.png'))
    
    # 可视化Grad-CAM结果（使用验证集中的一张图像）并保存热力图
    val_dataset = dataloaders['val'].dataset
    img_path, _ = val_dataset.samples[0]  # 使用验证集的第一张图像
    visualize_grad_cam(model_ft, img_path, class_names, save_path=os.path.join(log_dir, 'grad_cam_heatmap.png'))
    
    # 过拟合欠拟合分析
    analyze_fitting(train_losses, val_losses, train_accs, val_accs, log_dir)

    # 保存模型
    torch.save(model_ft.state_dict(), 'resnet50_finetuned.pth')
    print("模型已保存为: resnet50_finetuned.pth")