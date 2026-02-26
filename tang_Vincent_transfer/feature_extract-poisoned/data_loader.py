from torchvision import datasets, transforms
import torch

def load_data(data_folder, batch_size, train, kwargs):
    transform = {
        'train': transforms.Compose(
            [transforms.Resize([256, 256]),
                transforms.RandomCrop(224), # 隨機裁剪，為增加訓練資料多樣性
                transforms.RandomHorizontalFlip(), # 水平翻轉(依預設概率)，為增加訓練資料多樣性
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])]),
        'test': transforms.Compose(
            [transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])
        }
    data = datasets.ImageFolder(root = data_folder, transform=transform['train' if train else 'test'])
    
    # shuffle(要不要打亂數據順序), kwargs是定義維number_workers，drop_last:True表示假設剩下不完全的batch丟棄
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, **kwargs, drop_last = True if train else False) 
    return data_loader
 
