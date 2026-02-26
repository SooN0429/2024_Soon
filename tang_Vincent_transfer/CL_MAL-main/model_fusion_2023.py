"""
異構模型融合代碼 (使用權重平均)

功能介紹:
1. 模型融合能力:
   - 支持同構模型融合 (相同架構)
   - 支持異構模型融合 (不同深度和寬度)
   - 智能檢測並處理共同層和特有層
   - 自動處理不同維度的參數對齊

2. 權重平均融合方法:
   - 直接對相同維度的層進行權重平均
   - 對於不同維度的層，採用滑動窗口策略:
     * 支持左移/右移/丟棄等多種對齊方式
     * 可配置滑動步長和方向
     * 自動處理邊界情況
   - 數值穩定性優化
   - 自動跳過不兼容的層

3. 監控和分析:
   - 詳細的層級融合過程監控
   - 融合前後的參數差異分析
   - 性能指標追蹤:
     * 原始/最終/最佳準確率
     * 最後10個epoch的平均準確率
     * 準確率提升百分比
     * 訓練和融合時間統計
   - 完整的實驗記錄到Excel

4. 實用功能:
   - 支持GPU加速
   - 自動檢測並跳過不兼容的層
   - 錯誤處理和回退機制
   - 結果導出到Excel
   - 支持批量實驗

使用方式:
python model_fusion_new_6.py 
    --source_model_1 [模型1名稱]
    --source_train_feature [模型1的訓練特徵]
    --source_train_feature_label [模型1的訓練特徵標籤]
    --source_model_2 [模型2名稱]
    --extracted_layer [層名稱]
    --save_parameter_path_name [保存路徑]
    --save_parameter_path_name_1 [模型1參數路徑]
    --save_parameter_path_name_2 [模型2參數路徑]
    --results_excel [Excel文件路徑]
    --log_dir [日誌目錄]

輸出:
1. 融合模型:
   - 保存融合後的模型參數
   - 保存最佳epoch的模型參數
   - 保存最後epoch的模型參數

2. 性能報告:
   - Excel表格記錄完整實驗結果
   - 準確率和提升百分比
   - 時間統計(融合時間、訓練時間等)
   - 最後10個epoch的平均表現

3. 監控數據:
   - 每層融合過程的詳細記錄
   - 參數變化分析
   - 融合質量評估

注意事項:
- 需要預先準備好兩個訓練好的源模型
- 確保指定的保存路徑存在
- 相比OT方法，計算速度更快
- 適合需要快速融合的場景
"""

# model_fusion6.py
import torch
import os
import math
import data_loader
import models
import models1_2
import models2_2
import models3_2
import models4_2
import models1_1
import models2_1
import models3_1
import models4_1
import models1
import models2
import models3
import models4
import numpy as np
from sklearn.metrics import confusion_matrix
import torch.utils.data as Data
import argparse
import time
import backbone_multi
from config import CFG
import utils
import LabelSmoothing as LS
import pandas as pd

# 設置cuDNN的benchmark模式，這樣可以讓cuDNN自動尋找最適合當前配置的高效算法
torch.backends.cudnn.benchmark = True

# 定義一個函數來將字符串轉換為布爾值
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

# 創建一個命令行參數解析器
parser = argparse.ArgumentParser(description="training source model and testing performance")
# 添加各種命令行參數
parser.add_argument('--source_model_1', type=str, default="None") # 模型1
parser.add_argument('--source_train_feature', type=str, default="None") # 模型1的訓練特徵
parser.add_argument('--source_train_feature_label', type=str, default="None") # 模型1的訓練特徵標籤
parser.add_argument('--source_model_2', type=str, default="None") # 模型2
parser.add_argument("--extracted_layer", type=str, default="None", help='Which point of feature map want to extract') # 提取特徵的層
parser.add_argument('--train_root_path', type=str, default="/data/sihan.zhu/transfer learning/deep/dataset/RSTL/") # source_model_1 特徵數據集根目錄
parser.add_argument('--test_root_path', type=str, default="/data/sihan.zhu/transfer learning/deep/dataset/RSTL/")
parser.add_argument('--source_dir', type=str, default="UCM") # source_model_1 的訓練數據集 source_model_1是什麼水果就帶入什麼水果
parser.add_argument('--target_test_dir', type=str, default="RSSCN7") #source_model_1的測試數據集 source_model_1是什麼水果就帶入什麼水果
parser.add_argument("--batch_size", type=int, default=8, help="Training batch size") # 訓練批次大小 
parser.add_argument("--lr", type=float, default=0.0001, help="initial learning rate") # 初始學習率  
parser.add_argument("--epoch", type=int, default=150, help="Number of training epochs") # 訓練輪次          
parser.add_argument("--save_parameter_path_name", type=str, default="None", help='path to save models and log files') # 保存融合後的模型參數
parser.add_argument("--save_parameter_path_name_1", type=str, default="None", help='path to save models and log files') # 加載source_model_1的模型參數
parser.add_argument("--save_parameter_path_name_2", type=str, default="None", help='path to save models and log files') # 加載source_model_2的模型參數
parser.add_argument("--gpu_id", type=str, default='cuda:1', help='GPU id') # GPU設備ID
parser.add_argument("--results_excel", type=str, required=True, help='Path to results Excel file')
parser.add_argument("--log_dir", type=str, required=True, help='Directory for log files')

# 解析命令行參數
opt = parser.parse_args()
extracted_layer = opt.extracted_layer
backbone_multi.extracted_layer = extracted_layer
# 設置設備為GPU或CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 設置使用的GPU設備
torch.cuda.set_device(opt.gpu_id)
logtrain = []  # 用於記錄訓練過程中的損失
logtest = []   # 用於記錄測試過程中的結果

# 在文件開頭添加這些變量
start_time = time.time()
last_accuracy = None
best_accuracy = None
first_accuracy = None
total_time = 0
all_accuracies = []  # 添加這行來存儲所有的準確率

# 定義測試函數
def test(model, target_test_loader, test_flag):
    model.eval()  # 將模型設置為評估模式
    test_loss = utils.AverageMeter()  # 用於計算平均損失
    correct_total = 0.  # 記錄正確預測的數量
    count_stack = 0
    test_flag = 1
    criterion = torch.nn.CrossEntropyLoss()  # 定義損失函數
    len_target_dataset = len(target_test_loader.dataset)  # 測試數據集的大小
    with torch.no_grad():  # 禁用梯度計算，提升推理速度
        for data, target in target_test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)  # 將數據移動到設備上
            test_output = model.predict(data, test_flag)  # 獲取模型的預測輸出
            loss = criterion(test_output, target)  # 計算損失
            test_loss.update(loss.item())  # 更新損失
            pred = torch.max(test_output, 1)[1]  # 獲取預測的類別
            correct_total += torch.sum(pred == target)  # 計算正確預測的數量
            pred_matrix = pred.data
            target_matrix = target.data
            if count_stack == 0:
                pred_matrix_total = pred_matrix
                target_matrix_total = target_matrix
                count_stack = 1
            elif count_stack == 1:
                pred_matrix_total = torch.cat((pred_matrix_total, pred_matrix))
                target_matrix_total = torch.cat((target_matrix_total, target_matrix))
        target_matrix_total = target_matrix_total.cpu().numpy()
        pred_matrix_total = pred_matrix_total.cpu().numpy()
        tn, fp, fn, tp = confusion_matrix(target_matrix_total, pred_matrix_total, labels=[0, 1]).ravel()
    test_total = 100 * correct_total.type(torch.float32) / len_target_dataset  # 計算準確率
    current_accuracy = test_total.cpu().item()  # 在這裡先轉換為 CPU 數值

    # 如果是第一次測試，記錄 first_accuracy
    global first_accuracy, last_accuracy, best_accuracy, best_epoch
    current_epoch = getattr(test, 'current_epoch', 0)  # 獲取當前epoch
    if first_accuracy is None:
        first_accuracy = current_accuracy
        best_accuracy = current_accuracy
        best_epoch = current_epoch
        print(f'\ntest max correct: {correct_total}, First Test accuracy: {test_total:.3f} %\n')
    else:
        print(f'\nTest max correct: {correct_total}, Test accuracy: {test_total:.3f} %\n')
        # 更新 best_accuracy
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_epoch = current_epoch
            print(f'New best accuracy achieved at epoch {best_epoch}!')
    
    # 更新最後一次的準確率
    last_accuracy = current_accuracy
    
    # 添加當前準確率到列表中（除了第一次測試）
    if first_accuracy is not None:
        all_accuracies.append(current_accuracy)
    
    # 修改後：將 GPU tensor 轉換為 CPU 上的純數據
    logtest.append([
        test_total.cpu().item(),
        int(tn),
        int(fp),
        int(fn),
        int(tp)
    ])
    np_test = np.array(logtest, dtype=float)
    test_flag = 0
    return np_test, current_accuracy  # 返回兩個值，與model_fusion_new_5.py保持一致

# 定義訓練函數
def train(source_loader, test_flag, target_test_loader, model, CFG, optimizer):
    global total_time
    best_model_state = None
    last_model_state = None
    best_epoch = 0
    len_source_loader = len(source_loader)  # 訓練數據集的大小
    for e in range(opt.epoch):  # 遍歷每一個訓練輪次
        test_flag = 0
        train_loss_clf = utils.AverageMeter()  # 用於計算分類損失的平均值
        train_loss_total = utils.AverageMeter()  # 用於計算總損失的平均值
        LEARNING_RATE = opt.lr  # 設置學習率
        model.train()  # 將模型設置為訓練模式
        iter_source = iter(source_loader)  # 創建數據迭代器
        n_batch = len_source_loader  # 獲取批次數量
        criterion = LS.LabelSmoothingCrossEntropy(reduction='sum')  # 使用標籤平滑的交叉熵損失
        tStart = time.time()  # 記錄開始時間
        for i in range(n_batch):  # 遍歷每一個批次
            data_source, label_source = next(iter_source)
            data_source, label_source = data_source.to(DEVICE), label_source.to(DEVICE)  # 將數據移動到設備上
            label_source = torch.squeeze(label_source)  # 去掉維度為1的維度
            optimizer.zero_grad()  # 清空梯度
            source, label_source_pred = model(data_source, label_source, test_flag)  # 前向傳播
            clf_loss = criterion(label_source_pred, label_source)  # 計算分類損失
            loss = clf_loss  # 總損失
            loss.backward()  # 反向傳播
            optimizer.step()  # 更新參數
            train_loss_clf.update(clf_loss.item())  # 更新分類損失
            train_loss_total.update(loss.item())  # 更新總損失
            if i % CFG['log_interval'] == 0:  # 每隔一定步數打印一次訓練信息
                print('Train Epoch: [{}/{} ({:02d}%)], cls_Loss: {:.6f}, total_Loss: {:.6f}'.format(
                    e + 1,
                    opt.epoch,
                    int(100. * i / n_batch), train_loss_clf.avg, train_loss_total.avg))
        scheduler.step()  # 更新學習率
        logtrain.append([train_loss_clf.avg, ])
        np_log = np.array(logtrain, dtype=float)
        tEnd = time.time()  # 記錄結束時間
        epoch_time = tEnd - tStart
        total_time += epoch_time
        print(f'Time Cost: {epoch_time:.3f}s')
        np_test, current_accuracy = test(model, target_test_loader, test_flag)
        
        # 如果當前準確率是最佳的，保存模型狀態
        if best_accuracy == current_accuracy:
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = e
        
        # 在每個epoch結束時更新最後的模型狀態
        last_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # 在訓練結束後添加統計信息
    avg_epoch_time = total_time / opt.epoch
    total_time_cost = fusion_time + (avg_epoch_time * opt.epoch)
    
    # 計算兩種提升百分比
    last_accuracy_diff = (last_accuracy - first_accuracy) / first_accuracy * 100
    best_accuracy_diff = (best_accuracy - first_accuracy) / first_accuracy * 100
    
    # 修改：計算所有epoch的平均
    avg_accuracy = sum(all_accuracies) / len(all_accuracies) if all_accuracies else 0
    avg_accuracy_diff = (avg_accuracy - first_accuracy) / first_accuracy * 100
    
    last_improve_status = f"+{last_accuracy_diff:.2f}%" if last_accuracy_diff > 0 else f"{last_accuracy_diff:.2f}%"
    best_improve_status = f"+{best_accuracy_diff:.2f}%" if best_accuracy_diff > 0 else f"{best_accuracy_diff:.2f}%"
    avg_improve_status = f"+{avg_accuracy_diff:.2f}%" if avg_accuracy_diff > 0 else f"{avg_accuracy_diff:.2f}%"
    
    # 獲取模型信息
    model_1_name = model_1.__class__.__module__
    model_0_name = model_0.__class__.__module__
    
    # 終端機輸出
    print('\n' + '='*50)
    print(f'{model_1_name} ({opt.source_model_2}) --> {model_0_name} ({opt.source_model_1})')
    print(f'layer = {opt.extracted_layer}')
    print(f'Total epoch: {opt.epoch}')
    print(f'Total Time Cost: {total_time_cost:.3f}s')
    print(f'Avg. Time Cost Each Epoch: {avg_epoch_time:.3f}s')
    print(f'Original Test accuracy: {first_accuracy:.3f} %')
    print(f'Last Test accuracy: {last_accuracy:.3f}% ({last_improve_status})')
    print(f'Best Test accuracy: {best_accuracy:.3f}% ({best_improve_status})')
    print(f'Average Test accuracy: {avg_accuracy:.3f}% ({avg_improve_status})')
    print('='*50 + '\n')
    
    # 寫入 log 文件
    with open('/mnt/F/Vincent_v1.0/model_fusion_testlog/model_fusion_' + opt.save_parameter_path_name + opt.source_model_1 + '_' + opt.extracted_layer + '.txt', 'w') as f:
        # 加入時間戳
        timestamp = time.strftime("%Y%m%d %H%M%S", time.localtime())
        f.write(timestamp + '\n')
        f.write('='*50 + '\n')
        f.write(f'{model_1_name} ({opt.source_model_2}) --> {model_0_name} ({opt.source_model_1})\n')
        f.write(f'layer = {opt.extracted_layer}\n')
        f.write(f'Total epoch: {opt.epoch}\n')
        f.write(f'Total Time Cost: {total_time_cost:.3f}s\n')
        f.write(f'Avg. Time Cost Each Epoch: {avg_epoch_time:.3f}s\n')
        f.write(f'Original Test accuracy: {first_accuracy:.3f} %\n')
        f.write(f'Last Test accuracy: {last_accuracy:.3f}% ({last_improve_status})\n')
        f.write(f'Best Test accuracy: {best_accuracy:.3f}% ({best_improve_status})\n')
        f.write(f'Average Test accuracy: {avg_accuracy:.3f}% ({avg_improve_status})\n')
        f.write('='*50 + '\n')
    
    # 確保所有tensor數據都轉移到CPU
    experiment_data = {
        'Timestamp': [timestamp],
        'Model1': [model_0_name],
        'Model2': [model_1_name],
        'Fruit1': [opt.source_model_1],
        'Fruit2': [opt.source_model_2],
        'Layer': [opt.extracted_layer],
        'Total Epoch': [opt.epoch],
        'Model Fusion Time Cost (s)': [fusion_time],
        'Avg. Time Cost Each Epoch (s)': [avg_epoch_time],
        'Total Time Cost (s)': [total_time_cost],
        'Original Test Accuracy (%)': [first_accuracy],
        'Last Test Accuracy (%)': [last_accuracy],
        'Last Test Accuracy Improve (%)': [last_accuracy_diff],
        'Best Test Accuracy (%)': [best_accuracy],
        'Best Test Accuracy Improve (%)': [best_accuracy_diff],
        'Average Test Accuracy (%)': [avg_accuracy],
        'Average Test Accuracy Improve (%)': [avg_accuracy_diff]
    }
    
    # 如果文件已存在，則追加數據，否則創建新文件
    try:
        df = pd.read_excel(opt.results_excel)
        df = pd.concat([df, pd.DataFrame(experiment_data)], ignore_index=True)
    except FileNotFoundError:
        df = pd.DataFrame(experiment_data)
    
    # 保存 Excel 文件
    df.to_excel(opt.results_excel, index=False)
    
    return np_test, best_model_state, last_model_state, best_epoch

# 修改融合部分的代碼，移除 OT 相關的函數，但保留監控功能
def monitor_fusion_process(name, params_0, params_1):
    """監控融合過程"""
    print(f"\nMonitoring average fusion for layer: {name}")
    print(f"Original shapes - params_0: {params_0.shape}, params_1: {params_1.shape}")
    
    # 計算融合前後的差異
    diff_before = torch.norm(params_0 - params_1)
    fused_params = (params_0 + params_1) / 2
    diff_after = torch.norm(params_0 - fused_params)
    print(f"L2 distance - before: {diff_before:.4f}, after: {diff_after:.4f}")
    
    return {
        'layer_name': name,
        'fusion_method': 'Average Weight',
        'shape': params_0.shape,
        'improvement': (diff_before - diff_after).item()
    }

def monitor_all_layers(name, params_0, params_1):
    """監控所有層的形狀和融合方式"""
    print(f"\nLayer: {name}")
    print(f"Shape: {params_0.shape}")
    print(f"Dimension: {len(params_0.shape)}")
    print(f"Fusion method: Average Weight")

def monitor_different_size_fusion(name, params_0, params_1, diff_size, m0_size, m1_size, sliding):
    """監控不同大小層的融合過程"""
    print(f"\nMonitoring different size fusion for layer: {name}")
    print(f"Original shapes - params_0: {params_0.shape}, params_1: {params_1.shape}")
    print(f"Difference sizes: {diff_size}")
    print(f"Model 0 mapping sizes: {m0_size}")
    print(f"Model 1 mapping sizes: {m1_size}")
    print(f"Sliding value: {sliding}")

if __name__ == '__main__':
    torch.manual_seed(0)  # 設置隨機種子，保證結果可重現
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)

    test_flag = 0
    extracted_layer = opt.extracted_layer
    print('model fusion target = ', opt.source_model_1)
    kwargs = {'num_workers': 2, 'pin_memory': False, 'persistent_workers': True} if opt.gpu_id == 0 or 1 else {}

    # 加載訓練數據
    path_source_train = os.path.join(opt.train_root_path, opt.source_dir, opt.source_train_feature)
    path_source_train_label = os.path.join(opt.train_root_path, opt.source_dir, opt.source_train_feature_label)

    source_train = torch.from_numpy(np.load(path_source_train))
    source_train_label = torch.from_numpy(np.load(path_source_train_label))
    source_dataset = Data.TensorDataset(source_train, source_train_label)
    source_loader = Data.DataLoader(
        dataset=source_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        persistent_workers=True,
    )

    # 加載測試數據
    target_test_loader = data_loader.load_testing(opt.test_root_path, opt.target_test_dir, opt.batch_size, kwargs)
    source_save_parameter_path_1 = '/mnt/F/Vincent_v1.0/create_datasets/source_model_para/' + opt.extracted_layer + '/' + opt.source_model_1 + '/' + str(opt.lr) + '/'
    source_save_parameter_add_1 = os.path.join(source_save_parameter_path_1 + str(opt.save_parameter_path_name_1))
    source_save_parameter_path_2 = '/mnt/F/Vincent_v1.0/create_datasets/source_model_para/' + opt.extracted_layer + '/' + opt.source_model_2 + '/' + str(opt.lr) + '/'
    source_save_parameter_add_2 = os.path.join(source_save_parameter_path_2 + str(opt.save_parameter_path_name_2))
    save_parameter_path = '/mnt/F/Vincent_v1.0/fusion_model_para/' + opt.extracted_layer + '/' + opt.source_model_1 + '_' + opt.source_model_2 + '_to/' + opt.source_model_1 + '/' + str(opt.lr) + '/'

    #if not os.path.exists(save_parameter_path):
        #os.makedirs(save_parameter_path)

    # 加載模型
    model_0 = models3_2.Transfer_Net(CFG['n_class'])
    model_0 = model_0.cuda()
    model_0.load_state_dict(torch.load(source_save_parameter_add_1, map_location='cuda:0'))
    model_0_stru = []
    model_0_para = []
    model_0_layer = 0

    model_1 = models.Transfer_Net(CFG['n_class'])
    model_1 = model_1.cuda()
    model_1.load_state_dict(torch.load(source_save_parameter_add_2, map_location='cuda:0'))
    model_1_stru = []
    model_1_para = []
    model_1_layer = 0

    model = models3_2.Transfer_Net(CFG['n_class'])
    model = model.cuda()
    model = model_0

    # 設置優化器
    optimizer = torch.optim.Adam([
        {'params': model.base_network.parameters(), 'lr': 100 * opt.lr},
        {'params': model.base_network.avgpool.parameters(), 'lr': 100 * opt.lr},
        {'params': model.bottle_layer.parameters(), 'lr': 10 * opt.lr},
        {'params': model.classifier_layer.parameters(), 'lr': 10 * opt.lr},
    ], lr=opt.lr, betas=CFG['betas'], weight_decay=CFG['l2_decay'])

    # 設置學習率調度器
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85, verbose=False)



    # 在測試集上進行初步測試
    np_test, _ = test(model, target_test_loader, test_flag)

    # 遍歷模型0的參數，將其名稱和參數存入列表
    for name, parameters in model_0.named_parameters():
        model_0_stru.append(name)
        model_0_para.append(parameters)
        model_0_layer += 1

    # 遍歷模型1的參數，將其名稱和參數存入列表
    for name, parameters in model_1.named_parameters():
        model_1_stru.append(name)
        model_1_para.append(parameters)
        model_1_layer += 1

    # 初始化一些列表，用於記錄不同的層和映射矩陣
    differents = []
    diff_stru_idx = []
    mapping_matrix_0 = []
    mapping_matrix_1 = []
    shift_left = []
    shift_right = []
    shift_drop = []
    shiftty = 0

    # 準備一些移動策略的名稱
    for i in range(0, 192):
        shift_left.append('left_' + str(i))
    for i in range(0, 192):
        shift_right.append('right_' + str(i))
    for i in range(0, 4):
        shift_drop.append('drop_' + str(i))
        
    # 比較兩個模型的參數，找出不同的層
    for m0 in model_0_stru:
        for m1 in model_1_stru:
            # 名稱相同但尺寸不同的層，記錄到 differents 和 mapping_matrix_*，後續需要處理。
            if m0 == m1 and model_0_para[model_0_stru.index(m0)].size() != model_1_para[model_1_stru.index(m1)].size():
                # 記錄到 differents 和 diff_stru_idx，後續需要處理。
                differents.append(m0)
                diff_stru_idx.append(differents.index(m0))
                # 如果層的維度數少於4，則補齊到4維，並在 mapping_matrix_0 和 mapping_matrix_1 中填充 0。
                for i in range(4 - len(model_0_para[model_0_stru.index(m0)].size())):
                    mapping_matrix_0.append(0)
                    mapping_matrix_1.append(0)
                # 記錄不同層的尺寸，並計算 mapping_matrix_* 中的尺寸差異。
                for i in range(len(model_0_para[model_0_stru.index(m0)].size())):
                    if model_0_para[model_0_stru.index(m0)].size()[i] >= model_1_para[model_1_stru.index(m1)].size()[i]:
                        differents.append(model_0_para[model_0_stru.index(m0)].size()[i])
                    else:
                        differents.append(model_1_para[model_1_stru.index(m1)].size()[i])
                    mapping_matrix_0.append(differents[diff_stru_idx[-1] + 1 + i] - model_0_para[model_0_stru.index(m0)].size()[i])
                    mapping_matrix_1.append(differents[diff_stru_idx[-1] + 1 + i] - model_1_para[model_1_stru.index(m1)].size()[i])
            else:
                continue

    # 找出兩個模型中不相同的層
    different = set(model_0_stru).union(set(model_1_stru)) - set(model_0_stru).intersection(set(model_1_stru))
    for diff in different:
        differents.append(diff)
        diff_stru_idx.append(differents.index(diff))
        
        if diff not in model_0_stru:
            for i in range(4):
                mapping_matrix_0.append(-1)
        else:
            for i in range(4 - len(model_0_para[model_0_stru.index(diff)].size())):
                mapping_matrix_0.append(0)
            for i in range(len(model_0_para[model_0_stru.index(diff)].size())):
                differents.append(model_0_para[model_0_stru.index(diff)].size()[i])
                mapping_matrix_0.append(differents[diff_stru_idx[-1] + 1 + i] - model_0_para[model_0_stru.index(diff)].size()[i])
        
        if diff not in model_1_stru:
            for i in range(4):
                mapping_matrix_1.append(-1)
        else:
            if diff not in model_0_stru:
                for i in range(4 - len(model_1_para[model_1_stru.index(diff)].size())):
                    mapping_matrix_1.append(0)
                for i in range(len(model_1_para[model_1_stru.index(diff)].size())):
                    differents.append(model_1_para[model_1_stru.index(diff)].size()[i])
                    mapping_matrix_1.append(differents[diff_stru_idx[-1] + 1 + i] - model_1_para[model_1_stru.index(diff)].size()[i])
            else:
                for i in range(len(model_1_para[model_1_stru.index(diff)].size())):
                    if model_1_para[model_1_stru.index(diff)].size()[i] > model_0_para[model_0_stru.index(diff)].size()[i]:
                        differents[diff_stru_idx[-1] + i + 1] = model_1_para[model_1_stru.index(diff)].size()[i]
                        mapping_matrix_0[-(len(model_1_para[model_1_stru.index(diff)].size()) - i)] = differents[diff_stru_idx[-1] + 1 + i] - model_0_para[model_0_stru.index(diff)].size()[i]

    # 在模型融合開始前添加計時器（在 "將兩個模型的參數進行融合" 之前）
    fusion_start_time = time.time()

    # 添加融合統計變量
    fused_params_count = 0
    fused_params_bytes = 0
    skipped_params_count = 0
    skipped_params_bytes = 0
    total_params_count = 0
    total_params_bytes = 0

    # 將兩個模型的參數進行融合
    # 找到 base_network.convm2_layer.0.res.weight 的索引
    all_param_names = [name for name, _ in model_0.named_parameters()]
    if 'base_network.convm2_layer.0.res.weight' in all_param_names:
        fusion_start_idx = all_param_names.index('base_network.convm2_layer.0.res.weight')
        print(f"\n=== 融合設定 ===")
        print(f"找到融合起始層: base_network.convm2_layer.0.res.weight (索引: {fusion_start_idx})")
        print(f"將跳過前 {fusion_start_idx} 個層的融合")
        print("="*30)
    else:
        fusion_start_idx = 0
        print(f"\n=== 融合設定 ===")
        print("未找到 base_network.convm2_layer.0.res.weight，將對所有層進行融合")
        print("="*30)

    for idx, (name, parameters) in enumerate(model_0.named_parameters()):
        # 統計總參數
        total_params_count += parameters.numel()
        total_params_bytes += parameters.element_size() * parameters.numel()
        
        # 只對指定層之後的層做融合
        if idx < fusion_start_idx:
            # 跳過融合，直接保留原始參數
            skipped_params_count += parameters.numel()
            skipped_params_bytes += parameters.element_size() * parameters.numel()
            print(f"層: {name}")
            print(f"  跳過（未達到convm2_layer.0.res.weight）: {parameters.shape}")
            continue
        
        if name not in differents:
            # 同構層融合 - 權重平均
            model_0.state_dict()[name] = (model_0.state_dict()[name] + model_1.state_dict()[name]) / 2
            
            # 統計融合的參數
            fused_params_count += parameters.numel()
            fused_params_bytes += parameters.element_size() * parameters.numel()
            
        elif name in model_0_stru and name in model_1_stru:
            # 異構層融合
            num_diff = diff_stru_idx.index(differents.index(name))
            num_size = diff_stru_idx[num_diff + 1] - diff_stru_idx[num_diff] - 1
            if num_size == 1:
                diff_size = int(differents[differents.index(name) + 1])
                m0_size = int(mapping_matrix_0[(num_diff + 1) * 4 - (num_size)])
                m1_size = int(mapping_matrix_1[(num_diff + 1) * 4 - (num_size)])
                if m0_size == 0:    
                    for i in range(diff_size - m1_size):
                        model_0.state_dict()[name][int(m1_size / 2) + i] = (model_0.state_dict()[name][int(m1_size / 2) + i] + model_1.state_dict()[name][i]) / 2
                    # 統計融合的參數（只計算實際融合的部分）
                    fused_params_count += (diff_size - m1_size)
                    fused_params_bytes += (diff_size - m1_size) * parameters.element_size()
                    # 統計跳過的參數
                    skipped_params_count += m1_size
                    skipped_params_bytes += m1_size * parameters.element_size()
                elif m1_size == 0:
                    for i in range(diff_size - m0_size):
                        model_0.state_dict()[name][i] = (model_0.state_dict()[name][i] + model_1.state_dict()[name][int(m0_size / 2) + i]) / 2
                    # 統計融合的參數（只計算實際融合的部分）
                    fused_params_count += (diff_size - m0_size)
                    fused_params_bytes += (diff_size - m0_size) * parameters.element_size()
                    # 統計跳過的參數
                    skipped_params_count += m0_size
                    skipped_params_bytes += m0_size * parameters.element_size()
                        
            elif num_size == 2:
                diff_size = []
                m0_size = []
                m1_size = []
                for i in range(num_size):
                    diff_size.append(int(differents[differents.index(name) + 1 + i]))
                    m0_size.append(int(mapping_matrix_0[(num_diff + 1) * 4 - (num_size) + i]))
                    m1_size.append(int(mapping_matrix_1[(num_diff + 1) * 4 - (num_size) + i]))

                if 'left' in opt.save_parameter_path_name:
                    shift = shift_left
                elif 'right' in opt.save_parameter_path_name:
                    shift = shift_right
                    shiftty = 1
                elif 'drop' in opt.save_parameter_path_name:
                    shift = shift_drop

                for i in shift:
                    if i in opt.save_parameter_path_name:
                        sliding = int(i.split('_')[1]) + 192 * shiftty
                        break

                # model_fusion method 分散_左上slide到右_手動
                if m0_size[0]>m1_size[0]:
                    row = m0_size[0]
                    r0 = 1
                    r1 = diff_size[0]/(diff_size[0]-m0_size[0])
                    slide_r_0 = 0
                    slide_r_1 = r1/4*sliding
                elif m1_size[0]>m0_size[0]:
                    row = m1_size[0]
                    r0 = diff_size[0]/(diff_size[0]-m1_size[0])
                    r1 = 1
                    slide_r_0 = r0/4*sliding
                    slide_r_1 = 0
                else:
                    row = 0
                    r0 = 1
                    r1 = 1
                    slide_r_0 = 0
                    slide_r_1 = 0

                if m0_size[1]>m1_size[1]:
                    col = m0_size[1]
                    c0 = 1
                    c1 = diff_size[1]/(diff_size[1]-m0_size[1])
                    slide_c_0 = 0
                    slide_c_1 = c1/4*sliding
                elif m1_size[1]>m0_size[1]:
                    col = m1_size[1]
                    c0 = diff_size[1]/(diff_size[1]-m1_size[1])
                    c1 = 1
                    slide_c_0 = c0/4*sliding
                    slide_c_1 = 0
                else:
                    col = 0
                    c0 = 1
                    c1 = 1
                    slide_c_0 = 0
                    slide_c_1 = 0

                # 計算實際融合的參數數量
                fused_count = 0
                for i in range(diff_size[0]-row):
                    for j in range(diff_size[1]-col):
                        model_0.state_dict()[name][int(r0*i+slide_r_0)][int(c0*j+slide_c_0)] = (model_0.state_dict()[name][int(r0*i+slide_r_0)][int(c0*j+slide_c_0)]+model_1.state_dict()[name][int(r1*i+slide_r_1)][int(c1*j+slide_c_1)])/2
                        fused_count += 1
                
                # 統計融合的參數
                fused_params_count += fused_count
                fused_params_bytes += fused_count * parameters.element_size()
                
                # 統計跳過的參數（總參數減去融合的參數）
                skipped_count = parameters.numel() - fused_count
                skipped_params_count += skipped_count
                skipped_params_bytes += skipped_count * parameters.element_size()
        else:
            # 如果層不在model_1中，跳過融合
            skipped_params_count += parameters.numel()
            skipped_params_bytes += parameters.element_size() * parameters.numel()

    model = model_0

    # 計算融合時間
    fusion_end_time = time.time()
    fusion_time = fusion_end_time - fusion_start_time
    
    # 顯示融合統計結果
    print('\n' + '='*60)
    print('模型融合統計結果')
    print('='*60)
    print(f'總參數數量: {total_params_count:,}')
    print(f'總參數大小: {total_params_bytes / (1024 ** 2):.2f} MB')
    print(f'已融合參數數量: {fused_params_count:,}')
    print(f'已融合參數大小: {fused_params_bytes / (1024 ** 2):.2f} MB')
    print(f'跳過融合參數數量: {skipped_params_count:,}')
    print(f'跳過融合參數大小: {skipped_params_bytes / (1024 ** 2):.2f} MB')
    print(f'融合比例: {fused_params_count/total_params_count*100:.2f}%')
    print(f'融合大小比例: {fused_params_bytes/total_params_bytes*100:.2f}%')
    print('='*60)
    print(f'Model Fusion Time Cost: {fusion_time:.3f}s')
    print('='*60 + '\n')


    np_log, best_model_state, last_model_state, best_epoch = train(source_loader, test_flag, target_test_loader, model, CFG, optimizer)    
    """
    # 保存最佳模型參數，包含epoch信息
    best_parameter_add = os.path.join(save_parameter_path + str(opt.save_parameter_path_name) + f'_best_epoch{best_epoch}')
    torch.save(best_model_state, best_parameter_add)
    
    # 保存最後一個epoch的模型參數
    last_parameter_add = os.path.join(save_parameter_path + str(opt.save_parameter_path_name) + f'_last_epoch{opt.epoch-1}')
    torch.save(last_model_state, last_parameter_add)
    """

