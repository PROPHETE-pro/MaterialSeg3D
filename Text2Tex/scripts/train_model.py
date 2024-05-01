import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import os
from PIL import Image

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = os.listdir(data_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.images[idx])
        image = Image.open(img_name).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label_name = os.path.join('/data/zeyu_li/DMS/DMS_v1/labels/train', self.images[idx].replace(".jpg", ".png"))
        label = Image.open(label_name).convert("L")  # 使用标签图像的灰度通道作为整数标签
        if self.transform:
            label = self.transform(label)

        return image, label

# 数据目录
data_dir = '/data/zeyu_li/DMS/DMS_v1/images/train'

# 数据转换
data_transforms = transforms.Compose([
    transforms.Resize((768, 1024)),  # 手动调整图像大小以匹配模型的期望输入尺寸
    transforms.ToTensor(),
])

# 创建自定义数据集
image_dataset = CustomDataset(data_dir, transform=data_transforms)

# 创建数据加载器
data_loader = DataLoader(image_dataset, batch_size=32, shuffle=True)

# 3. 定义模型
model = models.resnet18(pretrained=True)
num_classes = 46  # 46 种材质分类
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

# 4. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 5. 训练模型
for epoch in range(10):  # 例如，训练 10 个周期
    running_loss = 0.0
    for i, data in enumerate(data_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(data_loader)}')

print('Training Finished')

# 6. 保存模型
model_save_path = '/data/zeyu_li/DMS/output/model.pth'
torch.save(model.state_dict(), model_save_path)
