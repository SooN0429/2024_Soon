# model_fusion_new_10_3.py繼承過來的，要做大量實驗
# 新嘗試 20250302
# 同構融合 統計對齊融合(雙向對齊到中間點) 融合比例可由參數 --fusion_alpha 調整
# 異構融合 使用 特徵空間相似性 對齊 成功對齊融合第一層bottle_layer.0.weight和bottle_layer.0.bias
# 異構融合 輸出表示空間對齊 對齊 成功對齊融合最後一層bottle_layer.x.weight和bottle_layer.x.bias

# 恢復回 0303的版本，0306版本輸出的結果與0303版本有較大的差異
# 改善尾層融合的方法，先使用投影矩陣將目標模型權重投影到源模型空間，再基於相似度進行動態加權融合 0309的版本
# 頭尾層均加入相似度動態調整，但融合比例還是要偏向"好"模型多一點，不然提升幅度有限 0310的版本
# 加入alpha和beta參數，分別控制同構層融合比例和異構層融合比例 0314的版本 想法是 aplpha=0.5 beta=0.3(偏向保留多一點"好模型"的參數) 還未測試(爛)
# 加入alpha和beta參數，0321的版本 想法是 aplpha=0.3 beta=0.5
# 0314 建議"只融合同構層"的實驗，請使用"model_fusion_new_10_1_1.py"這個版本的
# 0323 大全套
# 0326 加入比例參數，用於控制訓練樣本的使用比例 (0.0-1.0)
# 0331 感覺要加入當"bottle layer"一樣的時候（原本都當異構層的），是不是也要比照同構層辦理，用統計對齊融合？ 還沒改！
# 0402 算是現在功能最完整的版本
# 同構對齊融合只對convm2層進行
# 0618 加入"新的"統計對齊融合（參考: REPAIR、RESCALE），並加入adaptive_method參數，用於自動選擇統計對齊方法
# 0000 加入conv_M2層的通道相似性對齊功能（基於特徵統計的快速相似性匹配）


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
import torch.nn.functional as F
import logging
import torch.nn as nn
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy.optimize import linear_sum_assignment


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
parser.add_argument("--fusion_alpha", type=float, default=0.5, help='Fusion weight for homogeneous layers (between 0 and 1)') # 同構層融合權重參數
parser.add_argument("--fusion_beta", type=float, default=0.5, help='Fusion weight for heterogeneous layers (between 0 and 1)') # 添加異構層融合權重參數
parser.add_argument("--fusion_layers", type=str, default="all", choices=["all", "head_only", "tail_only", "homogeneous_only", "head_and_tail_only"], 
                    help='選擇要融合的層類型: all=所有層, head_only=僅頭層, tail_only=僅尾層, homogeneous_only=僅同構層(共享區塊), head_and_tail_only=僅頭尾層(個人化區塊)') # 添加融合層選擇參數
parser.add_argument("--train_ratio", type=float, default=1.0, 
                    help="比例參數，用於控制訓練樣本的使用比例 (0.0-1.0)")
parser.add_argument("--statistical_method", type=str, default="repair", 
                    choices=["repair", "rescale", "original"],
                    help="統計對齊方法選擇: repair=REPAIR方法(同時校正均值和標準差), rescale=RESCALE方法(僅校正標準差), original=原始方法(全局統計)")# original=原始方法(全局統計)是學姊的作法
parser.add_argument("--adaptive_method", type=str2bool, default=False,
                    help="是否根據層類型自動選擇統計對齊方法")
parser.add_argument("--channel_similarity", type=str2bool, default=True,
                    help="是否在統計對齊前進行通道相似性對齊")
parser.add_argument("--similarity_top_k", type=float, default=0.3,
                    help="通道相似性對齊中，選擇最相似的top_k比例進行重新排列")

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
last_accuracy = None  # 添加 last_accuracy
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
    return np_test, current_accuracy

def compute_correlation(covariance, eps=1e-7):
    """計算相關係數矩陣"""
    std = torch.diagonal(covariance).sqrt()
    covariance = covariance / (torch.clamp(torch.outer(std, std), min=eps))
    return covariance

def remove_col(x, idx, temp=None):
    """從矩陣中移除指定列"""
    if temp is None:
        return torch.cat([x[:, :idx], x[:, idx+1:]], dim=-1)
    else:
        R, C = x.shape
        temp = temp[:R, :C]
        _, L = x[:, idx+1:].shape
        temp[:, :L] = x[:, idx+1:]
        x[:, idx:idx+L] = temp[:, :L]
        return x[:, :C-1]

def match_tensors(metric, model_dims, a=0.3, b=0.125, print_merges=False):
    """
    匹配算法的簡化版本
    """
    if "covariance" in metric:
        sims = compute_correlation(metric["covariance"])
    
    O = sims.shape[0]
    remainder = max(model_dims)
    permutation_matrix = torch.eye(O, device=sims.device)
    temp_ = torch.empty_like(permutation_matrix).to(permutation_matrix)
    
    torch.diagonal(sims)[:] = -torch.inf
    
    num_models = len(model_dims)
    original_model = torch.zeros(O, device=sims.device).long()
    cur_dim = 0
    for i, dim in enumerate(model_dims):
        original_model[cur_dim:cur_dim + dim] = i
        cur_dim += dim
    
    to_remove = permutation_matrix.shape[1] - remainder
    budget = torch.zeros(num_models, device=sims.device).long()
    for i in range(num_models):
        budget[i] += int(to_remove * model_dims[i] // O * b + 1e-4)
    
    while permutation_matrix.shape[1] > remainder:
        best_idx = sims.reshape(-1).argmax()
        row_idx = best_idx % sims.shape[1]
        col_idx = best_idx // sims.shape[1]
        
        if col_idx < row_idx:
            row_idx, col_idx = col_idx, row_idx
        
        permutation_matrix[:, row_idx] += permutation_matrix[:, col_idx]
        permutation_matrix = remove_col(permutation_matrix, col_idx, temp=temp_)
        
        sims[:, row_idx] = torch.minimum(sims[:, row_idx], sims[:, col_idx])
        sims[:, row_idx] *= a
        sims = remove_col(sims, col_idx, temp=temp_)
        
        sims[row_idx, :] = torch.minimum(sims[row_idx, :], sims[col_idx, :])
        sims[row_idx, :] *= a
        sims = remove_col(sims.T, col_idx, temp=temp_).T
        
        row_origin, col_origin = original_model[row_idx], original_model[col_idx]
        original_model = remove_col(original_model[None, :], col_idx)[0]
        
        if row_origin == col_origin:
            origin = original_model[row_idx].item()
            budget[origin] -= 1
            
            if budget[origin] <= 0:
                selector = original_model == origin
                sims[selector[:, None] & selector[None, :]] = -torch.inf
    
    unmerge = permutation_matrix
    merge = permutation_matrix / (permutation_matrix.sum(dim=0, keepdim=True) + 1e-5)
    
    if print_merges:
        print("Final merge matrix shape:", merge.shape)
        print("Final unmerge matrix shape:", unmerge.shape)
    
    # 修改：不轉置merge矩陣
    return merge, unmerge

def align_heterogeneous_layers(source_layer, target_layer, beta=0.3):
    """
    進行異構層對齊
    
    Args:
        source_layer: 源模型層
        target_layer: 目標模型層
        beta: 控制異構層融合的權重參數（用於控制相似度衰減）
    """
    print("\n=== 開始異構層對齊 ===")
    print("Source weight shape:", source_layer.weight.shape)
    print("Target weight shape:", target_layer.weight.shape)
    
    # 檢測是否為尾層情況（輸出維度相同但輸入維度不同）
    is_tail_layer = source_layer.weight.shape[0] == target_layer.weight.shape[0] and source_layer.weight.shape[1] != target_layer.weight.shape[1]
    
    if is_tail_layer:
        print("\n檢測到尾層情況（輸出維度相同，輸入維度不同）")
        print("使用尾層神經元匹配方法...")
        
        # 為源模型和目標模型分別生成適合的隨機輸入
        batch_size = 1000
        source_input_dim = source_layer.weight.shape[1]
        target_input_dim = target_layer.weight.shape[1]
        source_random_input = torch.randn(batch_size, source_input_dim).to(source_layer.weight.device)
        target_random_input = torch.randn(batch_size, target_input_dim).to(target_layer.weight.device)
        
        print(f"Source random input shape: {source_random_input.shape}")
        print(f"Target random input shape: {target_random_input.shape}")
        
        # 計算輸出特徵
        with torch.no_grad():
            source_output = source_layer(source_random_input)
            target_output = target_layer(target_random_input)
        
        print("\n=== 特徵輸出維度 ===")
        print(f"Source output shape: {source_output.shape}")
        print(f"Target output shape: {target_output.shape}")
        
        try:
            # 新增：基於輸出特徵計算相似度矩陣（而不是直接用SVD）
            # 計算每個模型的協方差矩陣
            source_mean = source_output.mean(0, keepdim=True)
            target_mean = target_output.mean(0, keepdim=True)
            source_centered = source_output - source_mean
            target_centered = target_output - target_mean
            
            source_cov = torch.mm(source_centered.t(), source_centered) / (batch_size - 1)
            target_cov = torch.mm(target_centered.t(), target_centered) / (batch_size - 1)
            cross_cov = torch.mm(source_centered.t(), target_centered) / (batch_size - 1)
            
            print("\n=== 尾層協方差矩陣維度 ===")
            print("Source covariance shape:", source_cov.shape)
            print("Target covariance shape:", target_cov.shape)
            print("Cross covariance shape:", cross_cov.shape)
            
            # 構建完整的協方差矩陣
            full_cov = torch.cat([
                torch.cat([source_cov, cross_cov], dim=1),
                torch.cat([cross_cov.t(), target_cov], dim=1)
            ], dim=0)
            
            print("Full covariance shape:", full_cov.shape)
            
            # 計算相關性矩陣
            correlation = compute_correlation(full_cov)
            print("Correlation matrix shape:", correlation.shape)
            
            metric = {"covariance": correlation}
            # 注意：對於尾層，輸出維度相同，所以model_dims是相同的值
            model_dims = [source_layer.weight.size(0), target_layer.weight.size(0)]
            print("\n=== 模型維度信息 ===")
            print("Model dimensions:", model_dims)
            
            merge, unmerge = match_tensors(
                metric=metric,
                model_dims=model_dims,
                a=beta,
                b=0.125,
                print_merges=True
            )
            
            print("\n=== 合併矩陣維度 ===")
            print("Merge matrix shape:", merge.shape)
            print("Unmerge matrix shape:", unmerge.shape)
            
            # 步驟1：先使用SVD計算投影矩陣（與原來的方法相同）
            # 首先計算 W_target 的偽逆
            W_target_t = target_layer.weight.t()  # [target_dim, out_dim]
            W_target_pinv = torch.pinverse(target_layer.weight)  # [target_dim, out_dim]
            
            # 計算從目標輸入空間到源輸入空間的投影矩陣
            input_proj_matrix = torch.mm(W_target_pinv, source_layer.weight)  # [target_dim, source_dim]
            
            print(f"輸入投影矩陣形狀: {input_proj_matrix.shape}")
            print(f"這個矩陣將目標模型的權重投影到源模型的輸入空間")
            
            # 步驟2：使用投影矩陣轉換目標權重到源權重形狀
            aligned_target_weight = torch.mm(target_layer.weight, input_proj_matrix)  # [out_dim, source_dim]
            print(f"對齊後的目標權重形狀: {aligned_target_weight.shape}")
            
            # 步驟3：基於神經元匹配結果進行選擇性融合
            # 從合併矩陣中提取映射關係
            source_end = model_dims[0]
            A_source = merge[:, :source_end]  # [total_dims, source_dim]
            A_target = merge[:, source_end:]  # [total_dims, target_dim]
            
            print("\n=== 映射矩陣維度 ===")
            print("A_source (共享空間到源模型):", A_source.shape)
            print("A_target (共享空間到目標模型):", A_target.shape)
            
            # 計算合併權重
            fused_weight = torch.zeros_like(source_layer.weight)
            
            # 對於每個輸出神經元（對於尾層，輸出維度相同）
            for i in range(source_layer.weight.shape[0]):
                # 計算此神經元的融合權重
                source_weight = source_layer.weight[i]
                aligned_target_weight_i = aligned_target_weight[i]
                
                # 從相關性矩陣中獲取當前神經元的相似度值
                similarity = correlation[i, i + source_end]
                
                # 基於相似度進行動態加權融合
                # 相似度高的神經元給予更多的目標模型權重
                dynamic_beta = beta * (1.0 - similarity)  # 相似度越高，beta越小，目標權重佔比越大
                dynamic_beta = max(0.1, min(0.9, dynamic_beta))  # 限制在[0.1, 0.9]範圍內
                
                # 融合權重
                fused_weight[i] = dynamic_beta * source_weight + (1 - dynamic_beta) * aligned_target_weight_i
                
                if i % 10 == 0 or i == source_layer.weight.shape[0] - 1:  # 只打印一部分結果
                    print(f"神經元 {i}: 相似度 = {similarity:.4f}, 動態beta = {dynamic_beta:.4f}")
            
            print(f"融合後權重形狀: {fused_weight.shape}")
            
            # 處理偏置項
            fused_bias = None
            if source_layer.bias is not None and target_layer.bias is not None:
                # 對於尾層，輸出維度相同，可以直接融合偏置
                # 同樣可以使用相似度來調整融合比例
                fused_bias = torch.zeros_like(source_layer.bias)
                
                for i in range(source_layer.bias.shape[0]):
                    similarity = correlation[i, i + source_end]
                    dynamic_beta = beta * (1.0 - similarity)
                    dynamic_beta = max(0.1, min(0.9, dynamic_beta))
                    fused_bias[i] = dynamic_beta * source_layer.bias[i] + (1 - dynamic_beta) * target_layer.bias[i]
                
                print(f"融合後偏置形狀: {fused_bias.shape}")
            
            print("尾層神經元匹配融合完成")
            return fused_weight, fused_bias
                
        except Exception as tail_error:
            print(f"尾層神經元匹配融合失敗: {str(tail_error)}")
            print("嘗試使用原始方法進行尾層融合...")
            
            try:
                # 使用 SVD 計算輸出空間之間的映射關係
                U_source, S_source, V_source = torch.svd(source_output.t())
                U_target, S_target, V_target = torch.svd(target_output.t())
                
                # 計算目標模型到源模型的轉換矩陣（在輸出空間）
                transform_matrix = torch.mm(U_source, U_target.t())
                print(f"轉換矩陣形狀 (SVD方法): {transform_matrix.shape}")
                print(f"這個矩陣將目標模型的輸出空間映射到源模型的輸出空間")

                # 方法1：通過輸出特徵關係來推導權重變換矩陣
                # 對於尾層，我們需要將目標模型的權重投影到源模型的輸入空間
                
                # 首先計算 W_target 的偽逆
                W_target_t = target_layer.weight.t()  # [512, 128]
                W_target_pinv = torch.pinverse(target_layer.weight)  # [512, 128]
                
                # 計算從目標輸入空間到源輸入空間的投影矩陣
                input_proj_matrix = torch.mm(W_target_pinv, source_layer.weight)  # [512, 256]
                
                print(f"輸入投影矩陣形狀: {input_proj_matrix.shape}")
                print(f"這個矩陣將目標模型的權重投影到源模型的輸入空間")
                
                # 使用投影矩陣轉換目標權重到源權重形狀
                aligned_target_weight = torch.mm(target_layer.weight, input_proj_matrix)  # [128, 256]
                print(f"對齊後的目標權重形狀: {aligned_target_weight.shape}")
                
                # 融合權重
                fused_weight = beta * source_layer.weight + (1 - beta) * aligned_target_weight
                print(f"融合後權重形狀: {fused_weight.shape}")
                
                # 處理偏置項
                fused_bias = None
                if source_layer.bias is not None and target_layer.bias is not None:
                    # 對於尾層，輸出維度相同，可以直接融合偏置
                    fused_bias = beta * source_layer.bias + (1 - beta) * target_layer.bias
                    print(f"融合後偏置形狀: {fused_bias.shape}")
                
                print("使用原始方法尾層融合完成")
                return fused_weight, fused_bias
                
            except Exception as original_error:
                print(f"原始方法也失敗: {str(original_error)}")
                print("嘗試使用簡化方法進行尾層融合...")
                
                try:
                    # 簡化：直接使用SVD分解源權重和目標權重
                    U_s, S_s, V_s = torch.svd(source_layer.weight)
                    U_t, S_t, V_t = torch.svd(target_layer.weight)
                    
                    # 計算從目標到源的映射
                    V_t_truncated = V_t[:source_input_dim, :]
                    projection = torch.mm(V_s, V_t_truncated.t())
                    projected_weight = torch.mm(target_layer.weight, projection.t())
                    
                    # 融合權重
                    fused_weight = beta * source_layer.weight + (1 - beta) * projected_weight
                    
                    # 處理偏置
                    fused_bias = None
                    if source_layer.bias is not None and target_layer.bias is not None:
                        fused_bias = beta * source_layer.bias + (1 - beta) * target_layer.bias
                    
                    print("使用簡化方法成功融合尾層")
                    return fused_weight, fused_bias
                    
                except Exception as simplified_error:
                    print(f"簡化方法也失敗: {str(simplified_error)}")
                    print("保持源模型參數不變")
                    return source_layer.weight, source_layer.bias
        
        except Exception as svd_error:
            print(f"SVD計算失敗: {str(svd_error)}")
            print("保持源模型參數不變")
            return source_layer.weight, source_layer.bias
            
    else:
        # 非尾層情況的處理，保持原有代碼邏輯
        # 生成隨機輸入用於計算特徵
        batch_size = 1000
        input_dim = source_layer.weight.shape[1]
        random_input = torch.randn(batch_size, input_dim).to(source_layer.weight.device)
        print("Random input shape:", random_input.shape)
        
        # 計算輸出特徵
        with torch.no_grad():
            source_output = source_layer(random_input)
            target_output = target_layer(random_input)
        
        print("\n=== 特徵輸出維度 ===")
        print("Source output shape:", source_output.shape)
        print("Target output shape:", target_output.shape)
        
        # 計算每個模型的協方差矩陣
        source_mean = source_output.mean(0, keepdim=True)
        target_mean = target_output.mean(0, keepdim=True)
        source_centered = source_output - source_mean
        target_centered = target_output - target_mean
        
        source_cov = torch.mm(source_centered.t(), source_centered) / (batch_size - 1)
        target_cov = torch.mm(target_centered.t(), target_centered) / (batch_size - 1)
        cross_cov = torch.mm(source_centered.t(), target_centered) / (batch_size - 1)
        
        print("\n=== 協方差矩陣維度 ===")
        print("Source covariance shape:", source_cov.shape)
        print("Target covariance shape:", target_cov.shape)
        print("Cross covariance shape:", cross_cov.shape)
        
        # 構建完整的協方差矩陣
        full_cov = torch.cat([
            torch.cat([source_cov, cross_cov], dim=1),
            torch.cat([cross_cov.t(), target_cov], dim=1)
        ], dim=0)
        
        print("\n=== 完整協方差矩陣維度 ===")
        print("Full covariance shape:", full_cov.shape)
        
        # 計算相關性矩陣 (訓練free方法使用相關矩陣而非協方差矩陣)
        correlation = compute_correlation(full_cov)
        print("Correlation matrix shape:", correlation.shape)
        
        metric = {"covariance": correlation}
        model_dims = [source_layer.weight.size(0), target_layer.weight.size(0)]
        print("\n=== 模型維度信息 ===")
        print("Model dimensions:", model_dims)
        
        try:
            merge, unmerge = match_tensors(
                metric=metric,
                model_dims=model_dims,
                a=beta,
                b=0.125,
                print_merges=True
            )
            
            print("\n=== 合併矩陣維度 ===")
            print("Merge matrix shape:", merge.shape)
            print("Unmerge matrix shape:", unmerge.shape)
            
            # 檢查 merge 矩陣的形狀是否正確
            total_dims = sum(model_dims)  # source_dims + target_dims
            if merge.shape[0] != total_dims:
                print(f"警告: merge 矩陣行數 ({merge.shape[0]}) 與預期的 ({total_dims}) 不匹配")
                print("嘗試調整融合策略...")
            
            # 設置源模型和目標模型在合併矩陣中的列索引範圍
            source_start, source_end = 0, model_dims[0]
            target_start, target_end = model_dims[0], model_dims[0] + model_dims[1]
            
            # 修改: 從列方向提取轉換矩陣，而不是行方向
            A_source = merge[:, :source_end]  # 所有行，源模型的列 [total_dims, source_dim]
            A_target = merge[:, source_end:]  # 所有行，目標模型的列 [total_dims, target_dim]
            
            print("\n=== 映射矩陣維度 ===")
            print("A_source (共享空間到源模型):", A_source.shape)
            print("A_target (共享空間到目標模型):", A_target.shape)
            
            # 計算偽逆以獲得正確的映射矩陣
            # 轉置確保維度匹配
            A_source_t = A_source.t()  # [source_dim, total_dims]
            A_target_t = A_target.t()  # [target_dim, total_dims]
            
            # 計算轉換矩陣：從目標到源的映射
            # 使用 Moore-Penrose 偽逆進行計算
            source_to_target = torch.mm(A_target_t, torch.pinverse(A_source_t))  # [target_dim, source_dim]
            print("Source to target transformation matrix:", source_to_target.shape)
            
            # 轉換目標模型權重到源模型空間
            aligned_target_weight = torch.mm(source_to_target, source_layer.weight)  # [target_dim, input_dim]
            print("Target weight in source space:", aligned_target_weight.shape)
            
            # 融合權重
            # 確保維度匹配
            min_dim = min(source_layer.weight.size(0), aligned_target_weight.size(0))
            fused_weight = torch.zeros_like(source_layer.weight[:min_dim])
            
            # 新增：使用相似度動態調整融合權重
            source_end = model_dims[0]
            for i in range(min_dim):
                # 從相關性矩陣中獲取當前神經元的相似度值
                if i < source_end and i + source_end < correlation.shape[1]:
                    similarity = correlation[i, i + source_end]
                    
                    # 基於相似度進行動態加權融合
                    dynamic_beta = beta * (1.0 - similarity)  # 相似度越高，beta越小，目標權重佔比越大
                    dynamic_beta = max(0.1, min(0.9, dynamic_beta))  # 限制在[0.1, 0.9]範圍內
                    
                    # 融合權重
                    fused_weight[i] = dynamic_beta * source_layer.weight[i] + (1 - dynamic_beta) * aligned_target_weight[i]
                    
                    if i % 10 == 0 or i == min_dim - 1:  # 只打印一部分結果
                        print(f"頭層神經元 {i}: 相似度 = {similarity:.4f}, 動態beta = {dynamic_beta:.4f}")
                else:
                    # 如果索引超出範圍，使用原始beta
                    fused_weight[i] = beta * source_layer.weight[i] + (1 - beta) * aligned_target_weight[i]
            
            # 如果源模型維度大於目標模型，保留多餘部分
            if source_layer.weight.size(0) > min_dim:
                fused_weight = torch.cat([fused_weight, source_layer.weight[min_dim:]], dim=0)
                
            print("Fused weight shape:", fused_weight.shape)
            
            # 處理偏置項
            fused_bias = None
            if source_layer.bias is not None and target_layer.bias is not None:
                print("\n=== 偏置融合過程 ===")
                print("Source bias shape:", source_layer.bias.shape)
                print("Target bias shape:", target_layer.bias.shape)
                
                # 轉換目標模型偏置到源模型空間
                target_bias_in_source_space = torch.mm(source_to_target, source_layer.bias.unsqueeze(1)).squeeze(1)
                print("Target bias in source space:", target_bias_in_source_space.shape)
                
                # 融合偏置，確保維度匹配
                min_bias_dim = min(source_layer.bias.size(0), target_bias_in_source_space.size(0))
                fused_bias = torch.zeros_like(source_layer.bias[:min_bias_dim])
                
                # 使用相同的相似度動態調整偏置融合
                for i in range(min_bias_dim):
                    if i < source_end and i + source_end < correlation.shape[1]:
                        similarity = correlation[i, i + source_end]
                        dynamic_beta = beta * (1.0 - similarity)
                        dynamic_beta = max(0.1, min(0.9, dynamic_beta))
                        fused_bias[i] = dynamic_beta * source_layer.bias[i] + (1 - dynamic_beta) * target_bias_in_source_space[i]
                    else:
                        fused_bias[i] = beta * source_layer.bias[i] + (1 - beta) * target_bias_in_source_space[i]
                
                # 如果源模型偏置維度大於目標模型，保留多餘部分
                if source_layer.bias.size(0) > min_bias_dim:
                    fused_bias = torch.cat([fused_bias, source_layer.bias[min_bias_dim:]], dim=0)
                    
                print("Fused bias shape:", fused_bias.shape)
            
            return fused_weight, fused_bias
            
        except Exception as e:
            print(f"\n=== 異構融合失敗 ===")
            print(f"錯誤信息: {str(e)}")
            print("\n=== 錯誤診斷資訊 ===")
            
            # 打印出詳細的維度信息，幫助診斷問題
            if 'merge' in locals():
                print(f"merge矩陣形狀: {merge.shape}")
                total_dims = sum(model_dims)
                if total_dims != merge.shape[0]:
                    print(f"維度不匹配：total_dims ({total_dims}) != merge.shape[0] ({merge.shape[0]})")
            
            if 'A_source' in locals() and 'A_target' in locals():
                print(f"A_source形狀: {A_source.shape}")
                print(f"A_target形狀: {A_target.shape}")
            
            if 'source_to_target' in locals():
                print(f"source_to_target形狀: {source_to_target.shape}")
            
            if 'aligned_target_weight' in locals():
                print(f"aligned_target_weight形狀: {aligned_target_weight.shape}")
            
            # 嘗試備用方法
            print("\n嘗試備用融合方法...")
            try:
                # 使用 SVD 直接計算轉換矩陣
                with torch.no_grad():
                    # 計算目標模型到源模型的轉換矩陣
                    U_source, S_source, V_source = torch.svd(source_output.t())
                    U_target, S_target, V_target = torch.svd(target_output.t())
                    
                    # 取最小維度用於對齊
                    min_dim = min(source_layer.weight.size(0), target_layer.weight.size(0))
                    
                    # 創建轉換矩陣 (目標到源)
                    transform_matrix = torch.mm(U_source[:, :min_dim], U_target[:, :min_dim].t())
                    print("轉換矩陣 (SVD方法):", transform_matrix.shape)
                    
                    # 轉換目標權重到源空間
                    aligned_target_weight = torch.mm(transform_matrix, target_layer.weight)
                    print("對齊後的目標權重形狀:", aligned_target_weight.shape)
                    
                    # 計算相關性矩陣用於相似度動態調整
                    source_mean = source_output.mean(0, keepdim=True)
                    target_mean = target_output.mean(0, keepdim=True)
                    source_centered = source_output - source_mean
                    target_centered = target_output - target_mean
                    
                    source_cov = torch.mm(source_centered.t(), source_centered) / (source_centered.shape[0] - 1)
                    target_cov = torch.mm(target_centered.t(), target_centered) / (target_centered.shape[0] - 1)
                    cross_cov = torch.mm(source_centered.t(), target_centered) / (source_centered.shape[0] - 1)
                    
                    # 構建完整的協方差矩陣
                    full_cov = torch.cat([
                        torch.cat([source_cov, cross_cov], dim=1),
                        torch.cat([cross_cov.t(), target_cov], dim=1)
                    ], dim=0)
                    
                    # 計算相關性矩陣
                    correlation = compute_correlation(full_cov)
                    print("備用方法相關矩陣形狀:", correlation.shape)
                    
                    # 融合權重 (只使用共同維度)，使用相似度動態調整
                    min_dim = min(source_layer.weight.size(0), aligned_target_weight.size(0))
                    fused_weight = torch.zeros_like(source_layer.weight[:min_dim])
                    
                    for i in range(min_dim):
                        if i < source_layer.weight.size(0) and i < target_layer.weight.size(0) and i + target_layer.weight.size(0) < correlation.shape[1]:
                            similarity = correlation[i, i + target_layer.weight.size(0)]
                            dynamic_beta = beta * (1.0 - similarity)
                            dynamic_beta = max(0.1, min(0.9, dynamic_beta))
                            
                            fused_weight[i] = dynamic_beta * source_layer.weight[i] + (1 - dynamic_beta) * aligned_target_weight[i]
                            
                            if i % 10 == 0 or i == min_dim - 1:
                                print(f"備用方法頭層神經元 {i}: 相似度 = {similarity:.4f}, 動態beta = {dynamic_beta:.4f}")
                        else:
                            fused_weight[i] = beta * source_layer.weight[i] + (1 - beta) * aligned_target_weight[i]
                    
                    # 如果源模型維度大於目標模型，保留多餘部分
                    if source_layer.weight.size(0) > min_dim:
                        fused_weight = torch.cat([fused_weight, source_layer.weight[min_dim:]], dim=0)
                    
                    print("最終融合權重形狀:", fused_weight.shape)
                    
                    # 處理偏置項
                    fused_bias = None
                    if source_layer.bias is not None and target_layer.bias is not None:
                        aligned_target_bias = torch.mm(transform_matrix, target_layer.bias.unsqueeze(1)).squeeze(1)
                        
                        min_bias_dim = min(source_layer.bias.size(0), aligned_target_bias.size(0))
                        fused_bias = torch.zeros_like(source_layer.bias[:min_bias_dim])
                        
                        # 使用相同的相似度動態調整偏置融合
                        for i in range(min_bias_dim):
                            if i < source_layer.weight.size(0) and i < target_layer.weight.size(0) and i + target_layer.weight.size(0) < correlation.shape[1]:
                                similarity = correlation[i, i + target_layer.weight.size(0)]
                                dynamic_beta = beta * (1.0 - similarity)
                                dynamic_beta = max(0.1, min(0.9, dynamic_beta))
                                fused_bias[i] = dynamic_beta * source_layer.bias[i] + (1 - dynamic_beta) * aligned_target_bias[i]
                            else:
                                fused_bias[i] = beta * source_layer.bias[i] + (1 - beta) * aligned_target_bias[i]
                        
                        # 如果源模型偏置維度大於目標模型，保留多餘部分
                        if source_layer.bias.size(0) > min_bias_dim:
                            fused_bias = torch.cat([fused_bias, source_layer.bias[min_bias_dim:]], dim=0)
                    
                    return fused_weight, fused_bias
            except Exception as backup_error:
                print(f"備用融合方法也失敗: {str(backup_error)}")
                
            print("\n保持源模型參數不變")
            return source_layer.weight, source_layer.bias

# 智能選擇統計對齊方法的函數
def choose_statistical_method(layer_name, params_shape, base_method="repair"):
    """
    根據層類型和參數形狀智能選擇統計對齊方法
    
    參數:
        layer_name: 層名稱
        params_shape: 參數形狀
        base_method: 基礎方法選擇
        
    返回:
        method: 選擇的方法名稱
    """
    # 特別針對 conv_M2 層的優化選擇
    if 'convm2_layer' in layer_name.lower():
        # conv_M2 層是複雜的多分支卷積結構，需要特別處理
        if len(params_shape) == 4:  # 卷積權重 [out_channels, in_channels, h, w]
            # 根據conv_M2的複雜結構，使用REPAIR方法更好地保持特徵表示
            return "repair"
        elif len(params_shape) == 1:  # 偏置或BN參數
            if 'bias' in layer_name and 'bn' not in layer_name:
                # 卷積層的偏置
                return "rescale"
            elif 'bn' in layer_name and 'bias' in layer_name:
                # BatchNorm的偏置參數
                return "rescale"
            elif 'bn' in layer_name and 'weight' in layer_name:
                # BatchNorm的縮放參數
                return "repair"
            else:
                return "repair"
        else:
            return "repair"
    # 其他層的一般選擇邏輯
    elif 'conv' in layer_name.lower() or len(params_shape) == 4:
        # 一般卷積層：REPAIR方法更適合，因為需要保持通道間的關係
        return "repair"
    elif 'bn' in layer_name.lower() or 'batch' in layer_name.lower():
        # BatchNorm層：RESCALE方法更適合，避免破壞normalization的效果
        return "rescale"
    elif 'linear' in layer_name.lower() or 'fc' in layer_name.lower() or len(params_shape) == 2:
        # 全連接層：根據層的大小選擇
        if params_shape[0] > 512:  # 大型全連接層
            return "repair"  # 使用REPAIR保持更好的統計特性
        else:  # 小型全連接層
            return "rescale"  # 使用RESCALE避免過度校正
    elif len(params_shape) == 1:
        # 偏置向量：RESCALE方法更適合
        return "rescale"
    else:
        # 其他情況：使用基礎方法
        return base_method

def compute_channel_stats(tensor):
    """
    計算每個通道的統計特徵
    
    Args:
        tensor: 輸入張量 [out_channels, in_channels, h, w] 或 [out_channels, in_channels]
    
    Returns:
        stats: 統計特徵矩陣 [out_channels, 4] (均值、標準差、最大值、最小值)
    """
    if len(tensor.shape) == 4:  # 卷積層
        # 對每個輸出通道計算統計特徵
        mean_vals = torch.mean(tensor, dim=(1, 2, 3))  # [out_channels]
        std_vals = torch.std(tensor, dim=(1, 2, 3))    # [out_channels]
        max_vals = torch.max(torch.max(torch.max(tensor, dim=3)[0], dim=2)[0], dim=1)[0]  # [out_channels]
        min_vals = torch.min(torch.min(torch.min(tensor, dim=3)[0], dim=2)[0], dim=1)[0]  # [out_channels]
    elif len(tensor.shape) == 2:  # 全連接層
        # 對每個輸出神經元計算統計特徵
        mean_vals = torch.mean(tensor, dim=1)  # [out_features]
        std_vals = torch.std(tensor, dim=1)    # [out_features]
        max_vals = torch.max(tensor, dim=1)[0]  # [out_features]
        min_vals = torch.min(tensor, dim=1)[0]  # [out_features]
    else:
        raise ValueError(f"不支援的張量維度: {tensor.shape}")
    
    # 組合統計特徵 [channels, 4]
    stats = torch.stack([mean_vals, std_vals, max_vals, min_vals], dim=1)
    return stats

def compute_similarity_matrix(stats_0, stats_1):
    """
    計算兩組統計特徵之間的相似度矩陣
    
    Args:
        stats_0: 模型0的統計特徵 [channels_0, 4]
        stats_1: 模型1的統計特徵 [channels_1, 4]
    
    Returns:
        similarity_matrix: 相似度矩陣 [channels_0, channels_1]
    """
    # 正規化統計特徵
    stats_0_norm = F.normalize(stats_0, p=2, dim=1)  # L2正規化
    stats_1_norm = F.normalize(stats_1, p=2, dim=1)
    
    # 計算餘弦相似度矩陣
    similarity_matrix = torch.mm(stats_0_norm, stats_1_norm.t())  # [channels_0, channels_1]
    
    return similarity_matrix

def lightweight_channel_similarity_alignment(params_0, params_1, top_k_ratio=0.3):
    """
    輕量化通道相似性對齊（適用於conv_M2層）
    
    Args:
        params_0: 模型0的權重 [out_channels, in_channels, h, w] 或 [out_channels, in_channels]
        params_1: 模型1的權重（相同形狀）
        top_k_ratio: 選擇最相似的top_k比例進行重新排列
    
    Returns:
        aligned_params_1: 對齊後的模型1權重
        channel_mapping: 通道映射關係 (原始索引 -> 新索引)
    """
    print(f"  開始通道相似性對齊，top_k比例: {top_k_ratio}")
    
    # 步驟1：計算每個通道的統計特徵
    stats_0 = compute_channel_stats(params_0)  # [out_channels, 4]
    stats_1 = compute_channel_stats(params_1)  # [out_channels, 4]
    
    print(f"  統計特徵形狀: {stats_0.shape}")
    
    # 步驟2：計算相似度矩陣
    similarity_matrix = compute_similarity_matrix(stats_0, stats_1)  # [out_channels, out_channels]
    
    print(f"  相似度矩陣形狀: {similarity_matrix.shape}")
    print(f"  平均相似度: {torch.mean(similarity_matrix).item():.4f}")
    
    # 步驟3：選擇要重新排列的通道數量
    num_channels = params_0.shape[0]
    num_realign = int(num_channels * top_k_ratio)
    
    print(f"  總通道數: {num_channels}, 重新對齊通道數: {num_realign}")
    
    if num_realign == 0:
        print("  top_k_ratio太小，沒有通道需要重新對齊")
        return params_1, list(range(num_channels))
    
    # 步驟4：找到最相似的通道對
    similarity_flat = similarity_matrix.view(-1)
    _, top_indices = torch.topk(similarity_flat, num_realign)
    
    # 將一維索引轉換為二維索引
    row_indices = top_indices // num_channels  # 模型0的通道索引
    col_indices = top_indices % num_channels   # 模型1的通道索引
    
    # 步驟5：使用匈牙利算法找到最佳匹配
    # 轉換為成本矩陣（1 - 相似度）
    cost_matrix = (1 - similarity_matrix).cpu().numpy()
    
    # 限制匹配數量
    if num_realign < num_channels:
        # 只對最相似的部分進行匹配
        selected_rows = row_indices.cpu().numpy()
        selected_cols = col_indices.cpu().numpy()
        
        # 去重並取前num_realign個
        unique_rows = []
        unique_cols = []
        seen_rows = set()
        seen_cols = set()
        
        for r, c in zip(selected_rows, selected_cols):
            if r not in seen_rows and c not in seen_cols:
                unique_rows.append(r)
                unique_cols.append(c)
                seen_rows.add(r)
                seen_cols.add(c)
                if len(unique_rows) >= num_realign:
                    break
        
        # 對選中的子矩陣進行匈牙利算法
        if len(unique_rows) > 0:
            sub_cost_matrix = cost_matrix[np.ix_(unique_rows, unique_cols)]
            sub_row_ind, sub_col_ind = linear_sum_assignment(sub_cost_matrix)
            
            # 映射回原始索引
            optimal_rows = [unique_rows[i] for i in sub_row_ind]
            optimal_cols = [unique_cols[i] for i in sub_col_ind]
        else:
            optimal_rows, optimal_cols = [], []
    else:
        # 對整個矩陣進行匹配
        optimal_rows, optimal_cols = linear_sum_assignment(cost_matrix)
    
    # 步驟6：創建通道映射
    channel_mapping = list(range(num_channels))  # 初始映射：每個通道對應自己
    
    # 應用最佳匹配的重新排列
    for model_0_idx, model_1_idx in zip(optimal_rows, optimal_cols):
        channel_mapping[model_0_idx] = model_1_idx
    
    print(f"  實際重新排列的通道對數: {len(optimal_rows)}")
    
    # 步驟7：應用通道重新排列
    aligned_params_1 = params_1.clone()
    
    # 重新排列輸出通道
    for target_idx, source_idx in enumerate(channel_mapping):
        if source_idx != target_idx:  # 只有需要交換的才處理
            aligned_params_1[target_idx] = params_1[source_idx]
    
    # 計算對齊後的相似度提升
    original_similarity = torch.mean(torch.diag(similarity_matrix)).item()
    
    # 計算對齊後的相似度
    aligned_stats_1 = compute_channel_stats(aligned_params_1)
    aligned_similarity_matrix = compute_similarity_matrix(stats_0, aligned_stats_1)
    aligned_similarity = torch.mean(torch.diag(aligned_similarity_matrix)).item()
    
    print(f"  原始對角線相似度: {original_similarity:.4f}")
    print(f"  對齊後對角線相似度: {aligned_similarity:.4f}")
    print(f"  相似度提升: {aligned_similarity - original_similarity:.4f}")
    
    return aligned_params_1, channel_mapping

# 添加統計對齊融合函數
def statistical_alignment_fusion(params_0, params_1, alpha=0.5, eps=1e-5, repair_type='repair', layer_name="", 
                                enable_channel_similarity=False, similarity_top_k=0.3):
    """
    基於統計對齊融合（優化版本）
    
    參數:
        params_0: 第一個模型的參數
        params_1: 第二個模型的參數
        alpha: 融合權重，默認0.5表示兩個模型權重相等
        eps: 防止除零錯誤的小常數
        repair_type: 修復類型 ('repair', 'rescale', 'original')
        layer_name: 層名稱，用於調試輸出
        enable_channel_similarity: 是否啟用通道相似性對齊
        similarity_top_k: 通道相似性對齊的top_k比例
        
    返回:
        fused_params: 融合後的參數
    """
    # 步驟1：通道相似性對齊（如果啟用且為同構層）
    if enable_channel_similarity and params_0.shape == params_1.shape and len(params_0.shape) >= 2:
        print(f"  啟用通道相似性對齊")
        try:
            aligned_params_1, channel_mapping = lightweight_channel_similarity_alignment(
                params_0, params_1, top_k_ratio=similarity_top_k
            )
            params_1 = aligned_params_1  # 使用對齊後的參數進行後續統計融合
            print(f"  通道相似性對齊完成")
        except Exception as e:
            print(f"  通道相似性對齊失敗，使用原始參數: {str(e)}")
    
    # 步驟2：統計對齊融合（原有邏輯保持不變）
    # 如果選擇原始方法，保持向後相容性
    if repair_type == 'original':
        # 原始的全局統計對齊方法
        mean_0 = torch.mean(params_0)
        std_0 = torch.std(params_0)
        mean_1 = torch.mean(params_1) 
        std_1 = torch.std(params_1)
        
        mean_mid = (mean_0 + mean_1) / 2
        std_mid = (std_0 + std_1) / 2
        
        normalized_0 = (params_0 - mean_0) / (std_0 + eps)
        normalized_1 = (params_1 - mean_1) / (std_1 + eps)
        
        aligned_0 = normalized_0 * std_mid + mean_mid
        aligned_1 = normalized_1 * std_mid + mean_mid
        
        fused_params = alpha * aligned_0 + (1-alpha) * aligned_1
        return fused_params
    
    # 方法1: REPAIR - 同時校正均值和標準差（神經元級別）
    elif repair_type == 'repair':
        if len(params_0.shape) >= 2:  # 權重矩陣
            # 針對卷積層的特殊處理
            mean_0 = torch.mean(params_0, dim=(1,2,3), keepdim=True)  # 每個輸出通道的均值
            std_0 = torch.std(params_0, dim=(1,2,3), keepdim=True)
            mean_1 = torch.mean(params_1, dim=(1,2,3), keepdim=True)
            std_1 = torch.std(params_1, dim=(1,2,3), keepdim=True)
        else:  # 偏置向量
            mean_0 = torch.mean(params_0)
            std_0 = torch.std(params_0)
            mean_1 = torch.mean(params_1)
            std_1 = torch.std(params_1)
        
        # 計算目標統計特性
        mean_target = alpha * mean_0 + (1-alpha) * mean_1
        std_target = alpha * std_0 + (1-alpha) * std_1
        
        # 分別對齊兩個模型到目標統計特性
        normalized_0 = (params_0 - mean_0) / (std_0 + eps)
        normalized_1 = (params_1 - mean_1) / (std_1 + eps)
        
        aligned_0 = normalized_0 * std_target + mean_target
        aligned_1 = normalized_1 * std_target + mean_target
        
        # 最終融合
        fused_params = alpha * aligned_0 + (1-alpha) * aligned_1
        
    # 方法2: RESCALE - 僅校正標準差
    elif repair_type == 'rescale':
        if len(params_0.shape) >= 2:
            if len(params_0.shape) == 4:  # 卷積層
                std_0 = torch.std(params_0, dim=(1,2,3), keepdim=True)
                std_1 = torch.std(params_1, dim=(1,2,3), keepdim=True)
            else:  # 全連接層
                std_0 = torch.std(params_0, dim=1, keepdim=True)
                std_1 = torch.std(params_1, dim=1, keepdim=True)
        else:  # 偏置向量
            std_0 = torch.std(params_0)
            std_1 = torch.std(params_1)
            
        std_target = alpha * std_0 + (1-alpha) * std_1
        
        rescaled_0 = params_0 * std_target / (std_0 + eps)
        rescaled_1 = params_1 * std_target / (std_1 + eps)
        
        fused_params = alpha * rescaled_0 + (1-alpha) * rescaled_1
    
    else:
        raise ValueError(f"不支援的repair_type: {repair_type}")
    
    return fused_params

# 定義訓練函數
def train(source_loader, test_flag, target_test_loader, model, CFG, optimizer):
    global total_time
    best_model_state = None
    last_model_state = None
    best_epoch = 0
    len_source_loader = len(source_loader)
    
    for e in range(opt.epoch):
        test.current_epoch = e  # 設置當前epoch
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
    
    # 計算各種提升百分比
    last_accuracy_diff = (last_accuracy - first_accuracy) / first_accuracy * 100
    best_accuracy_diff = (best_accuracy - first_accuracy) / first_accuracy * 100
    
    # 修改：計算所有epoch的平均
    avg_accuracy = sum(all_accuracies) / len(all_accuracies) if all_accuracies else 0
    avg_accuracy_diff = (avg_accuracy - first_accuracy) / first_accuracy * 100
    
    last_improve_status = f"+{last_accuracy_diff:.2f}%" if last_accuracy_diff > 0 else f"{last_accuracy_diff:.2f}%"
    best_improve_status = f"+{best_accuracy_diff:.2f}%" if best_accuracy_diff > 0 else f"{best_accuracy_diff:.2f}%"
    avg_improve_status = f"+{avg_accuracy_diff:.2f}%" if avg_accuracy_diff > 0 else f"{avg_accuracy_diff:.2f}%"
    
    # 獲取模型資訊
    model_1_name = model_1.__class__.__module__
    model_0_name = model_0.__class__.__module__
    
    # 終端機輸出
    print('\n' + '='*50)
    print(f'{model_1_name} ({opt.source_model_2}) --> {model_0_name} ({opt.source_model_1})')
    print(f'layer = {opt.extracted_layer}')
    print(f'同構層融合權重 Alpha = {opt.fusion_alpha}')
    print(f'異構層融合權重 Beta = {opt.fusion_beta}')
    print(f'統計對齊方法 = {opt.statistical_method}')
    print(f'自適應方法選擇 = {opt.adaptive_method}')
    print(f'Total epoch: {opt.epoch}')
    print(f'Model Fusion Time Cost: {fusion_time:.3f}s')
    print(f'Avg. Time Cost Each Epoch: {avg_epoch_time:.3f}s')
    print(f'Total Time Cost: {total_time_cost:.3f}s')
    print(f'Original Test Accuracy (%)', [first_accuracy])
    print(f'Last Test Accuracy (%)', [last_accuracy])
    print(f'Last Test Accuracy Improve (%)', [last_accuracy_diff])
    print(f'Best Test Accuracy (%)', [best_accuracy])
    print(f'Best Test Accuracy Improve (%)', [best_accuracy_diff])
    print(f'Average Test Accuracy (all epochs) (%)', [avg_accuracy])
    print(f'Average Test Accuracy Improve (%)', [avg_accuracy_diff])
    
    # 獲取當前時間戳用於 Excel
    timestamp = time.strftime("%Y%m%d %H%M%S", time.localtime())
    
    # Excel 相關代碼
    experiment_data = {
        'Timestamp': [timestamp],
        'Model1': [model_0_name],
        'Model2': [model_1_name],
        'Fruit1': [opt.source_model_1],
        'Fruit2': [opt.source_model_2],
        'Layer': [opt.extracted_layer],
        'Alpha': [opt.fusion_alpha],  # 添加alpha值記錄
        'Beta': [opt.fusion_beta],  # 添加beta值記錄
        'Statistical Method': [opt.statistical_method],  # 添加統計方法記錄
        'Adaptive Method': [opt.adaptive_method],  # 添加自適應方法記錄
        'Total Epoch': [opt.epoch],
        'Model Fusion Time Cost (s)': [fusion_time],
        'Avg. Time Cost Each Epoch (s)': [avg_epoch_time],
        'Total Time Cost (s)': [total_time_cost],
        'Original Test Accuracy (%)': [first_accuracy],
        'Last Test Accuracy (%)': [last_accuracy],
        'Last Test Accuracy Improve (%)': [last_accuracy_diff],
        'Best Test Accuracy (%)': [best_accuracy],
        'Best Test Accuracy Improve (%)': [best_accuracy_diff],
        'Average Test Accuracy (all epochs) (%)': [avg_accuracy],
        'Average Test Accuracy Improve (%)': [avg_accuracy_diff]
    }
    
    # 確保目錄存在
    excel_dir = os.path.dirname(opt.results_excel)
    if not os.path.exists(excel_dir):
        os.makedirs(excel_dir)
        print(f"創建目錄: {excel_dir}")
    
    # 如果文件已存在，則追加數據，否則創建新文件
    try:
        df = pd.read_excel(opt.results_excel)
        df = pd.concat([df, pd.DataFrame(experiment_data)], ignore_index=True)
    except FileNotFoundError:
        df = pd.DataFrame(experiment_data)
    
    # 保存 Excel 文件
    df.to_excel(opt.results_excel, index=False)
    
    return np_test, best_model_state, last_model_state, best_epoch
         
  
def get_common_layers(model_0, model_1):
    """獲取兩個模型共同的層"""
    layers_0 = set(model_0.state_dict().keys())
    layers_1 = set(model_1.state_dict().keys())
    return layers_0.intersection(layers_1)

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
    if opt.train_ratio < 1.0:
        dataset_size = len(source_dataset)
        subset_size = int(dataset_size * opt.train_ratio)
        indices = torch.randperm(dataset_size)[:subset_size]
        source_loader = Data.DataLoader(
            dataset=source_dataset,
            batch_size=opt.batch_size,
            sampler=Data.SubsetRandomSampler(indices),
            num_workers=2,
            drop_last=True,
            persistent_workers=True
        )
    else:
        source_loader = Data.DataLoader(
            dataset=source_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=2,
            drop_last=True,
            persistent_workers=True
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
    # Target model
    model_0 = models1.Transfer_Net(CFG['n_class'])
    model_0 = model_0.cuda()
    model_0.load_state_dict(torch.load(source_save_parameter_add_1, map_location='cuda:0'))
    model_0_stru = []
    model_0_para = []
    model_0_layer = 0

    # Source model
    model_1 = models.Transfer_Net(CFG['n_class'])
    model_1 = model_1.cuda()
    model_1.load_state_dict(torch.load(source_save_parameter_add_2, map_location='cuda:0'))
    model_1_stru = []
    model_1_para = []
    model_1_layer = 0

    # Fused model
    model = models1.Transfer_Net(CFG['n_class'])
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

    
    for name, params_0 in model_0.state_dict().items():
        # 如果是BN層的追蹤參數，跳過
        if "num_batches_tracked" in name:
            continue
            
        if name not in model_1.state_dict():
            print(f"{name} | {params_0.shape} | 在model_1中未找到對應層")
            continue
            

    
    fusion_start_time = time.time()
    # 修改融合開始前的提示信息
    print("\n開始進行模型融合...")
    print(f"融合模式: {opt.fusion_layers}")

    if opt.fusion_layers == "all":
        print("對模型頭層和尾層進行異構融合，其他層使用統計對齊融合或保持原樣")
    elif opt.fusion_layers == "head_only":
        print("僅對頭層進行異構融合，其他層保持原有狀態")
    elif opt.fusion_layers == "tail_only":
        print("僅對尾層進行異構融合，其他層保持原有狀態")
    elif opt.fusion_layers == "head_and_tail_only":
        print("僅對頭層和尾層進行異構融合，其他層保持原有狀態")
    else:  # homogeneous_only
        print("僅對同構層進行融合，跳過所有異構層")

    # 檢測瓶頸層索引
    model_0_bottle_layers = [i for i, name in enumerate(model_0.bottle_layer) if isinstance(name, nn.Linear) or isinstance(name, nn.Conv2d)]
    model_1_bottle_layers = [i for i, name in enumerate(model_1.bottle_layer) if isinstance(name, nn.Linear) or isinstance(name, nn.Conv2d)]

    # 找出頭層和尾層索引
    model_0_first_layer_idx = model_0_bottle_layers[0] if model_0_bottle_layers else None
    model_1_first_layer_idx = model_1_bottle_layers[0] if model_1_bottle_layers else None
    model_0_last_layer_idx = model_0_bottle_layers[-1] if model_0_bottle_layers else None
    model_1_last_layer_idx = model_1_bottle_layers[-1] if model_1_bottle_layers else None

    print(f"模型0瓶頸層索引: {model_0_bottle_layers}, 最後一層: {model_0_last_layer_idx}")
    print(f"模型1瓶頸層索引: {model_1_bottle_layers}, 最後一層: {model_1_last_layer_idx}")

    # 1. 融合頭層
    if opt.fusion_layers in ["all", "head_only", "head_and_tail_only"] and model_0_first_layer_idx is not None and model_1_first_layer_idx is not None:
        print(f"\n=== 頭層融合 ===")
        print(f"嘗試融合: model_0.bottle_layer[{model_0_first_layer_idx}] 和 model_1.bottle_layer[{model_1_first_layer_idx}]")
        try:
            fused_weight, fused_bias = align_heterogeneous_layers(
                model_0.bottle_layer[model_0_first_layer_idx], 
                model_1.bottle_layer[model_1_first_layer_idx],
                beta=opt.fusion_beta
            )
            # 更新權重
            model.state_dict()[f'bottle_layer.{model_0_first_layer_idx}.weight'].data.copy_(fused_weight)
            if fused_bias is not None:
                model.state_dict()[f'bottle_layer.{model_0_first_layer_idx}.bias'].data.copy_(fused_bias)
            print("頭層異構融合完成")
        except Exception as e:
            print(f"頭層異構融合失敗，保持model_0原有參數: {e}")
    else:
        if opt.fusion_layers in ["all", "head_only", "head_and_tail_only"]:
            print("\n=== 頭層融合 ===")
            print("跳過頭層融合：未選擇融合頭層或至少一個模型沒有找到可用的頭層")
    
    # 2. 融合尾層
    if opt.fusion_layers in ["all", "tail_only", "head_and_tail_only"] and model_0_last_layer_idx is not None and model_1_last_layer_idx is not None:
        print(f"\n=== 尾層融合 ===")
        print(f"嘗試融合: model_0.bottle_layer[{model_0_last_layer_idx}] 和 model_1.bottle_layer[{model_1_last_layer_idx}]")
        try:
            fused_weight, fused_bias = align_heterogeneous_layers(
                model_0.bottle_layer[model_0_last_layer_idx],
                model_1.bottle_layer[model_1_last_layer_idx],
                beta=opt.fusion_beta
            )
            # 更新權重
            model.state_dict()[f'bottle_layer.{model_0_last_layer_idx}.weight'].data.copy_(fused_weight)
            if fused_bias is not None:
                model.state_dict()[f'bottle_layer.{model_0_last_layer_idx}.bias'].data.copy_(fused_bias)
            print("尾層異構融合完成")
        except Exception as e:
            print(f"尾層異構融合失敗，保持model_0原有參數: {e}")
    else:
        if opt.fusion_layers in ["all", "tail_only", "head_and_tail_only"]:
            print("\n=== 尾層融合 ===")
            print("跳過尾層融合：未選擇融合尾層或至少一個模型沒有找到可用的尾層")

    # 3. 處理其他層
    print("\n=== 處理其餘層 ===")
    
    # 首先掃描並顯示所有 conv_M2 相關的層
    conv_m2_layers = []
    for name in model_0.state_dict().keys():
        if 'base_network.convm2_layer' in name:
            conv_m2_layers.append(name)
    
    if conv_m2_layers:
        print(f"\n檢測到的conv_M2相關層: {conv_m2_layers}")
    else:
        print("\n警告：未檢測到任何conv_M2相關層！")
        print("可能的層名模式：")
        for name in list(model_0.state_dict().keys())[:10]:  # 顯示前10個層名作為參考
            print(f"  - {name}")
    
    for name, params_0 in model_0.state_dict().items():
        # 跳過所有的bottle_layer層 - 無論它們是否已被處理過
        if 'bottle_layer' in name:
            # 只顯示一次跳過信息，避免信息過多
            if '.weight' in name:
                print(f"\n層: {name}")
                print(f"跳過瓶頸層，保持原始參數: {params_0.shape}")
            continue
        
        # 如果是BN層的追蹤參數，直接跳過融合，保持原來的參數
        if "num_batches_tracked" in name:
            print(f"跳過 {name} 的融合，保持原始參數")
            continue
        
        if name not in model_1.state_dict():
            print(f"警告：在model_1中未找到 {name}，跳過此層的融合")
            continue
        
        # 獲取兩個模型的參數
        params_0 = params_0.data
        params_1 = model_1.state_dict()[name].data
        
        # 檢查參數形狀是否相同（是否為同構層）
        is_homogeneous = params_0.shape == params_1.shape
        
        # 根據融合模式決定是否融合
        if not is_homogeneous and opt.fusion_layers == "homogeneous_only":
            print(f"\n層: {name}")
            print(f"跳過異構層融合，保持原始參數: {params_0.shape} vs {params_1.shape}")
            continue
        
        # 如果是僅頭尾層模式，跳過其他層的融合
        if opt.fusion_layers == "head_and_tail_only":
            print(f"\n層: {name}")
            print(f"模式為僅頭尾層融合，跳過此層: {params_0.shape}")
            continue
        
        # 更精確地判斷是否為自定義卷積層conv_M2
        # 根據backbone_multi.py的架構，convm2_layer是包含conv_M2模塊的Sequential容器
        # 包含所有權重和偏置參數，但排除num_batches_tracked和running統計
        is_conv_m2_layer = ('base_network.convm2_layer' in name and 
                           ('weight' in name or 'bias' in name) and
                           'num_batches_tracked' not in name and
                           'running_mean' not in name and 
                           'running_var' not in name)
        
        # 同構層融合，但只融合自定義卷積層conv_M2
        if is_homogeneous:
            if is_conv_m2_layer:
                print(f"\n層: {name}")
                print(f"✓ conv_M2層統計對齊融合: {params_0.shape}")
                try:
                    # 選擇統計對齊方法
                    if opt.adaptive_method:
                        selected_method = choose_statistical_method(name, params_0.shape, opt.statistical_method)
                        print(f"  自動選擇方法: {selected_method}")
                    else:
                        selected_method = opt.statistical_method
                        print(f"  使用指定方法: {selected_method}")
                    
                    # 計算融合前的統計特性用於比較
                    before_mean_0 = torch.mean(params_0).item()
                    before_std_0 = torch.std(params_0).item()
                    before_mean_1 = torch.mean(params_1).item()
                    before_std_1 = torch.std(params_1).item()
                    
                    fused_params = statistical_alignment_fusion(
                        params_0, params_1, 
                        alpha=opt.fusion_alpha, 
                        eps=1e-5, 
                        repair_type=selected_method,
                        layer_name=name,
                        enable_channel_similarity=opt.channel_similarity,
                        similarity_top_k=opt.similarity_top_k
                    )
                    
                    # 計算融合後的統計特性
                    after_mean = torch.mean(fused_params).item()
                    after_std = torch.std(fused_params).item()
                    
                    model.state_dict()[name].data.copy_(fused_params)
                    
                    print(f"  融合前 M0: μ={before_mean_0:.4f}, σ={before_std_0:.4f}")
                    print(f"  融合前 M1: μ={before_mean_1:.4f}, σ={before_std_1:.4f}")
                    print(f"  融合後: μ={after_mean:.4f}, σ={after_std:.4f}")
                    
                except Exception as e:
                    print(f"  ✗ 融合失敗，保持model_0原有參數: {e}")
            else:
                print(f"\n層: {name}")
                print(f"  跳過非conv_M2層: {params_0.shape}")
        # 異構層融合（僅當選擇不是僅同構層時）
        elif opt.fusion_layers != "homogeneous_only":
            # 使用相同的判斷邏輯
            is_conv_m2_layer_hetero = ('base_network.convm2_layer' in name and 
                                      ('weight' in name or 'bias' in name) and
                                      'num_batches_tracked' not in name and
                                      'running_mean' not in name and 
                                      'running_var' not in name)
            
            if is_conv_m2_layer_hetero:
                print(f"\n層: {name}")
                print(f"✓ 異構conv_M2層部分融合: {params_0.shape} vs {params_1.shape}")
                try:
                    # 僅融合共同部分
                    min_dims = [min(d0, d1) for d0, d1 in zip(params_0.shape, params_1.shape)]
                    slice_obj = tuple(slice(0, d) for d in min_dims)
                    
                    # 創建與model_0參數相同形狀的張量
                    fused_params = params_0.clone()
                    
                    # 在共同維度上使用統計對齊融合
                    params_0_slice = params_0[slice_obj]
                    params_1_slice = params_1[slice_obj]
                    
                    # 選擇統計對齊方法
                    if opt.adaptive_method:
                        selected_method = choose_statistical_method(name, params_0_slice.shape, opt.statistical_method)
                        print(f"  自動選擇方法: {selected_method}")
                    else:
                        selected_method = opt.statistical_method
                        print(f"  使用指定方法: {selected_method}")
                    
                    # 計算融合前的統計特性
                    before_mean_0 = torch.mean(params_0_slice).item()
                    before_std_0 = torch.std(params_0_slice).item()
                    before_mean_1 = torch.mean(params_1_slice).item()
                    before_std_1 = torch.std(params_1_slice).item()
                    
                    fused_slice = statistical_alignment_fusion(
                        params_0_slice, params_1_slice, 
                        alpha=opt.fusion_alpha, 
                        eps=1e-5, 
                        repair_type=selected_method,
                        layer_name=f"{name}_conv_M2_slice",
                        enable_channel_similarity=opt.channel_similarity,
                        similarity_top_k=opt.similarity_top_k
                    )
                    
                    # 計算融合後的統計特性
                    after_mean = torch.mean(fused_slice).item()
                    after_std = torch.std(fused_slice).item()
                    
                    fused_params[slice_obj] = fused_slice
                    model.state_dict()[name].data.copy_(fused_params)
                    
                    print(f"  共同部分融合 {min_dims}")
                    print(f"  融合前 M0: μ={before_mean_0:.4f}, σ={before_std_0:.4f}")
                    print(f"  融合前 M1: μ={before_mean_1:.4f}, σ={before_std_1:.4f}")
                    print(f"  融合後: μ={after_mean:.4f}, σ={after_std:.4f}")
                    
                except Exception as e:
                    print(f"  ✗ conv_M2融合失敗，保持model_0原有參數: {e}")
            else:
                print(f"\n層: {name}")
                print(f"  跳過非conv_M2層: {params_0.shape}")

    model = model_0  # 這行需要保留，因為上面對model的修改可能發生在model_0的拷貝上

    # 計算融合時間並打印
    fusion_end_time = time.time()
    fusion_time = fusion_end_time - fusion_start_time
    print('\n' + '='*50)
    print(f'Model Fusion Time Cost: {fusion_time:.3f}s')
    print('='*50 + '\n')
    np_test, best_model_state, last_model_state, best_epoch = train(source_loader, test_flag, target_test_loader, model, CFG, optimizer)

    # 註釋掉保存模型的部分
    '''
    # 保存最佳模型參數，包含epoch信息
    best_parameter_add = os.path.join(save_parameter_path) + str(opt.save_parameter_path_name) + f'_best_epoch{best_epoch}'
    torch.save(best_model_state, best_parameter_add)
    
    # 保存最後一個epoch的模型參數
    last_parameter_add = os.path.join(save_parameter_path) + str(opt.save_parameter_path_name) + f'_last_epoch{opt.epoch-1}'
    torch.save(last_model_state, last_parameter_add)
    '''