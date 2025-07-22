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

# 设置随机种子确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# CBAM注意力机制实现
class ChannelAttention(nn.Module):
    """通道注意力模块"""
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """空间注意力模块"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)

class CBAM(nn.Module):
    """CBAM注意力模块：通道注意力+空间注意力"""
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_channels, reduction_ratio)  # 通道注意力模块
        self.spatial_att = SpatialAttention(kernel_size)  # 空间注意力模块
        
    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x

# 修改ResNet50的残差块，添加CBAM注意力机制
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)  # 保持输出尺寸不变

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    """带CBAM注意力机制的ResNet50残差块"""
    expansion = 4
    
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, use_cbam=True):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        
        # 残差路径
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        
        # CBAM注意力模块
        self.use_cbam = use_cbam
        if self.use_cbam:
            self.cbam = CBAM(planes * self.expansion)
            
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        # 应用CBAM注意力
        if self.use_cbam:
            out = self.cbam(out)
            
        # 处理下采样
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out

# 创建带CBAM注意力机制的ResNet50模型
def resnet50_cbam(pretrained=True, **kwargs):
    """Constructs a ResNet50 model with CBAM attention."""
    # 创建基础ResNet50模型
    model = models.resnet50(pretrained=pretrained)
    
    # 修改层结构，使用带CBAM的残差块
    layers = []
    
    # 第一个卷积层和池化层保持不变
    layers.append(model.conv1)
    layers.append(model.bn1)
    layers.append(model.relu)
    layers.append(model.maxpool)
    
    # 修改layer1-4，使用带CBAM的残差块
    # layer1不使用CBAM，保持原始结构
    layers.append(model.layer1)
    
    # layer2-4使用CBAM
    inplanes = 256
    block = Bottleneck
    planes = [64, 128, 256, 512]  # 每个layer的输出通道数
    num_blocks = [3, 4, 6, 3]  # 每个layer的残差块数量
    strides = [1, 2, 2, 2]  # 每个layer的下采样步长
    
    for i in range(1, 4):  # layer2-4
        layer_name = f'layer{i+1}'
        layer = []
        downsample = None
        
        if strides[i] != 1 or inplanes != planes[i] * block.expansion:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes[i] * block.expansion, strides[i]),
                nn.BatchNorm2d(planes[i] * block.expansion),
            )
            
        layer.append(block(inplanes, planes[i], strides[i], downsample))
        inplanes = planes[i] * block.expansion
        
        for _ in range(1, num_blocks[i]):
            layer.append(block(inplanes, planes[i]))
            
        # 创建Sequential模块并添加到layers列表
        layers.append(nn.Sequential(*layer))
    
    # 修改后的模型主体
    model_body = nn.Sequential(*layers)
    
    # 替换原始模型的层
    model.conv1 = model_body[0]
    model.bn1 = model_body[1]
    model.relu = model_body[2]
    model.maxpool = model_body[3]
    model.layer1 = model_body[4]
    model.layer2 = model_body[5]
    model.layer3 = model_body[6]
    model.layer4 = model_body[7]
    
    return model

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

# 可视化Grad-CAM结果
def visualize_grad_cam(model, img_path, class_names, target_layer='layer4.2.conv3'):
    """使用Grad-CAM可视化模型关注区域"""
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
    plt.show()

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
    return model

# 可视化模型预测结果
def visualize_model(model, dataloaders, class_names, num_images=6):
    """可视化模型的预测结果"""
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
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                
                # 将归一化的图像还原以便显示
                img = inputs.cpu().data[j].numpy().transpose((1, 2, 0))
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
                img = np.clip(img, 0, 1)
                plt.imshow(img)
                
                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        
        model.train(mode=was_training)

# 主函数
if __name__ == "__main__":
    # 设置数据集路径，需要根据实际情况修改
    data_dir = "resnet_data"
    
    # 加载数据
    dataloaders, dataset_sizes, class_names = load_data(data_dir)
    num_classes = len(class_names)
    print(f"训练集大小: {dataset_sizes['train']}, 验证集大小: {dataset_sizes['val']}")
    print(f"类别数量: {num_classes}, 类别名称: {class_names}")
    
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载带CBAM注意力机制的ResNet50模型
    model_ft = resnet50_cbam(pretrained=True)
    
    # 修改最后一层以适应我们的分类任务
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    
    # 将模型移到GPU
    model_ft = model_ft.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    
    # 学习率调度器
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    
    # 训练模型
    model_ft = train_model(
        model_ft, 
        criterion, 
        optimizer_ft, 
        exp_lr_scheduler, 
        dataloaders, 
        dataset_sizes,
        num_epochs=50
    )
    
     # 可视化Grad-CAM结果（使用验证集中的一张图像）
    val_dataset = dataloaders['val'].dataset
    val_dataset2 = dataloaders['train'].dataset
    img_path, _ = val_dataset.samples[0]  # 使用验证集的第一张图像
    img_path2, _ = val_dataset2.samples[0]


    visualize_grad_cam(model_ft, img_path, class_names)
    visualize_grad_cam(model_ft, img_path2, class_names)
    
    # 保存模型
    torch.save(model_ft.state_dict(), 'resnet50_cbam_finetuned.pth')
    print("模型已保存为: resnet50_cbam_finetuned.pth")    