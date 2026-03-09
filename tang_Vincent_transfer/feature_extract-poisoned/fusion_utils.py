# fusion_utils.py
# 自 model_fusion_2025.py 抽出的融合函式，僅依賴 torch / numpy / scipy，不依賴 models / backbone_multi。
# 供 model3class_fusion2025.py 等腳本使用。

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def compute_correlation(covariance, eps=1e-7):
    """計算相關係數矩陣"""
    std = torch.diagonal(covariance).sqrt()
    covariance = covariance / (torch.clamp(torch.outer(std, std), min=eps))
    return covariance


def remove_col(x, idx, temp=None):
    """從矩陣中移除指定列"""
    if temp is None:
        return torch.cat([x[:, :idx], x[:, idx + 1 :]], dim=-1)
    else:
        R, C = x.shape
        temp = temp[:R, :C]
        _, L = x[:, idx + 1 :].shape
        temp[:, :L] = x[:, idx + 1 :]
        x[:, idx : idx + L] = temp[:, :L]
        return x[:, : C - 1]


def match_tensors(metric, model_dims, a=0.3, b=0.125, print_merges=False):
    """匹配算法的簡化版本"""
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
        original_model[cur_dim : cur_dim + dim] = i
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

        row_origin, col_origin = original_model[row_idx].item(), original_model[col_idx].item()
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

    return merge, unmerge


def align_heterogeneous_layers(source_layer, target_layer, beta=0.3):
    """
    進行異構層對齊。
    source_layer: 寫入端（目標端 model_0）的層
    target_layer: 被融合端（來源端 model_1）的層
    beta: 控制異構層融合的權重參數
    返回: (fused_weight, fused_bias) 寫回 source_layer 對應的 state_dict。
    """
    print("\n=== 開始異構層對齊 ===")
    print("Source weight shape:", source_layer.weight.shape)
    print("Target weight shape:", target_layer.weight.shape)

    is_tail_layer = (
        source_layer.weight.shape[0] == target_layer.weight.shape[0]
        and source_layer.weight.shape[1] != target_layer.weight.shape[1]
    )

    if is_tail_layer:
        print("\n檢測到尾層情況（輸出維度相同，輸入維度不同）")
        batch_size = 1000
        source_input_dim = source_layer.weight.shape[1]
        target_input_dim = target_layer.weight.shape[1]
        source_random_input = torch.randn(batch_size, source_input_dim).to(source_layer.weight.device)
        target_random_input = torch.randn(batch_size, target_input_dim).to(target_layer.weight.device)

        with torch.no_grad():
            source_output = source_layer(source_random_input)
            target_output = target_layer(target_random_input)

        try:
            source_mean = source_output.mean(0, keepdim=True)
            target_mean = target_output.mean(0, keepdim=True)
            source_centered = source_output - source_mean
            target_centered = target_output - target_mean
            source_cov = torch.mm(source_centered.t(), source_centered) / (batch_size - 1)
            target_cov = torch.mm(target_centered.t(), target_centered) / (batch_size - 1)
            cross_cov = torch.mm(source_centered.t(), target_centered) / (batch_size - 1)
            full_cov = torch.cat(
                [
                    torch.cat([source_cov, cross_cov], dim=1),
                    torch.cat([cross_cov.t(), target_cov], dim=1),
                ],
                dim=0,
            )
            correlation = compute_correlation(full_cov)
            model_dims = [source_layer.weight.size(0), target_layer.weight.size(0)]
            merge, unmerge = match_tensors(metric={"covariance": correlation}, model_dims=model_dims, a=beta, b=0.125, print_merges=True)

            source_end = model_dims[0]
            W_target_pinv = torch.pinverse(target_layer.weight)
            input_proj_matrix = torch.mm(W_target_pinv, source_layer.weight)
            aligned_target_weight = torch.mm(target_layer.weight, input_proj_matrix)

            fused_weight = torch.zeros_like(source_layer.weight)
            for i in range(source_layer.weight.shape[0]):
                similarity = correlation[i, i + source_end]
                dynamic_beta = max(0.1, min(0.9, beta * (1.0 - similarity)))
                fused_weight[i] = dynamic_beta * source_layer.weight[i] + (1 - dynamic_beta) * aligned_target_weight[i]

            fused_bias = None
            if source_layer.bias is not None and target_layer.bias is not None:
                fused_bias = torch.zeros_like(source_layer.bias)
                for i in range(source_layer.bias.shape[0]):
                    similarity = correlation[i, i + source_end]
                    dynamic_beta = max(0.1, min(0.9, beta * (1.0 - similarity)))
                    fused_bias[i] = dynamic_beta * source_layer.bias[i] + (1 - dynamic_beta) * target_layer.bias[i]
            print("尾層神經元匹配融合完成")
            return fused_weight, fused_bias

        except Exception as tail_error:
            print(f"尾層神經元匹配融合失敗: {str(tail_error)}")
            try:
                W_target_pinv = torch.pinverse(target_layer.weight)
                input_proj_matrix = torch.mm(W_target_pinv, source_layer.weight)
                aligned_target_weight = torch.mm(target_layer.weight, input_proj_matrix)
                fused_weight = beta * source_layer.weight + (1 - beta) * aligned_target_weight
                fused_bias = None
                if source_layer.bias is not None and target_layer.bias is not None:
                    fused_bias = beta * source_layer.bias + (1 - beta) * target_layer.bias
                return fused_weight, fused_bias
            except Exception:
                try:
                    U_s, S_s, V_s = torch.svd(source_layer.weight)
                    U_t, S_t, V_t = torch.svd(target_layer.weight)
                    V_t_truncated = V_t[:source_input_dim, :]
                    projection = torch.mm(U_s, V_t_truncated.t())
                    projected_weight = torch.mm(target_layer.weight, projection.t())
                    fused_weight = beta * source_layer.weight + (1 - beta) * projected_weight
                    fused_bias = None
                    if source_layer.bias is not None and target_layer.bias is not None:
                        fused_bias = beta * source_layer.bias + (1 - beta) * target_layer.bias
                    return fused_weight, fused_bias
                except Exception:
                    return source_layer.weight.data.clone(), source_layer.bias.data.clone() if source_layer.bias is not None else None

    # 非尾層（頭層或同 shape）
    batch_size = 1000
    input_dim = source_layer.weight.shape[1]
    random_input = torch.randn(batch_size, input_dim).to(source_layer.weight.device)
    with torch.no_grad():
        source_output = source_layer(random_input)
        target_output = target_layer(random_input)

    source_mean = source_output.mean(0, keepdim=True)
    target_mean = target_output.mean(0, keepdim=True)
    source_centered = source_output - source_mean
    target_centered = target_output - target_mean
    source_cov = torch.mm(source_centered.t(), source_centered) / (batch_size - 1)
    target_cov = torch.mm(target_centered.t(), target_centered) / (batch_size - 1)
    cross_cov = torch.mm(source_centered.t(), target_centered) / (batch_size - 1)
    full_cov = torch.cat(
        [torch.cat([source_cov, cross_cov], dim=1), torch.cat([cross_cov.t(), target_cov], dim=1)],
        dim=0,
    )
    correlation = compute_correlation(full_cov)
    model_dims = [source_layer.weight.size(0), target_layer.weight.size(0)]

    try:
        merge, unmerge = match_tensors(metric={"covariance": correlation}, model_dims=model_dims, a=beta, b=0.125, print_merges=True)
        source_end = model_dims[0]
        A_source = merge[:, :source_end]
        A_target = merge[:, source_end:]
        A_source_t = A_source.t()
        A_target_t = A_target.t()
        source_to_target = torch.mm(A_target_t, torch.pinverse(A_source_t))
        aligned_target_weight = torch.mm(source_to_target, source_layer.weight)

        min_dim = min(source_layer.weight.size(0), aligned_target_weight.size(0))
        fused_weight = torch.zeros_like(source_layer.weight[:min_dim])
        for i in range(min_dim):
            if i < source_end and i + source_end < correlation.shape[1]:
                similarity = correlation[i, i + source_end]
                dynamic_beta = max(0.1, min(0.9, beta * (1.0 - similarity)))
                fused_weight[i] = dynamic_beta * source_layer.weight[i] + (1 - dynamic_beta) * aligned_target_weight[i]
            else:
                fused_weight[i] = beta * source_layer.weight[i] + (1 - beta) * aligned_target_weight[i]
        if source_layer.weight.size(0) > min_dim:
            fused_weight = torch.cat([fused_weight, source_layer.weight[min_dim:]], dim=0)

        fused_bias = None
        if source_layer.bias is not None and target_layer.bias is not None:
            target_bias_in_source_space = torch.mm(source_to_target, source_layer.bias.unsqueeze(1)).squeeze(1)
            min_bias_dim = min(source_layer.bias.size(0), target_bias_in_source_space.size(0))
            fused_bias = torch.zeros_like(source_layer.bias[:min_bias_dim])
            for i in range(min_bias_dim):
                if i < source_end and i + source_end < correlation.shape[1]:
                    similarity = correlation[i, i + source_end]
                    dynamic_beta = max(0.1, min(0.9, beta * (1.0 - similarity)))
                    fused_bias[i] = dynamic_beta * source_layer.bias[i] + (1 - dynamic_beta) * target_bias_in_source_space[i]
                else:
                    fused_bias[i] = beta * source_layer.bias[i] + (1 - beta) * target_bias_in_source_space[i]
            if source_layer.bias.size(0) > min_bias_dim:
                fused_bias = torch.cat([fused_bias, source_layer.bias[min_bias_dim:]], dim=0)
        return fused_weight, fused_bias

    except Exception as e:
        print(f"異構融合失敗: {str(e)}")
        try:
            U_source, S_source, V_source = torch.svd(source_output.t())
            U_target, S_target, V_target = torch.svd(target_output.t())
            min_dim = min(source_layer.weight.size(0), target_layer.weight.size(0))
            transform_matrix = torch.mm(U_source[:, :min_dim], U_target[:, :min_dim].t())
            aligned_target_weight = torch.mm(transform_matrix, target_layer.weight)
            fused_weight = torch.zeros_like(source_layer.weight[:min_dim])
            for i in range(min_dim):
                if i + target_layer.weight.size(0) < correlation.shape[1]:
                    similarity = correlation[i, i + target_layer.weight.size(0)]
                    dynamic_beta = max(0.1, min(0.9, beta * (1.0 - similarity)))
                    fused_weight[i] = dynamic_beta * source_layer.weight[i] + (1 - dynamic_beta) * aligned_target_weight[i]
                else:
                    fused_weight[i] = beta * source_layer.weight[i] + (1 - beta) * aligned_target_weight[i]
            if source_layer.weight.size(0) > min_dim:
                fused_weight = torch.cat([fused_weight, source_layer.weight[min_dim:]], dim=0)
            fused_bias = None
            if source_layer.bias is not None and target_layer.bias is not None:
                aligned_target_bias = torch.mm(transform_matrix, target_layer.bias.unsqueeze(1)).squeeze(1)
                min_bias_dim = min(source_layer.bias.size(0), aligned_target_bias.size(0))
                fused_bias = torch.zeros_like(source_layer.bias[:min_bias_dim])
                for i in range(min_bias_dim):
                    if i + target_layer.weight.size(0) < correlation.shape[1]:
                        similarity = correlation[i, i + target_layer.weight.size(0)]
                        dynamic_beta = max(0.1, min(0.9, beta * (1.0 - similarity)))
                        fused_bias[i] = dynamic_beta * source_layer.bias[i] + (1 - dynamic_beta) * aligned_target_bias[i]
                    else:
                        fused_bias[i] = beta * source_layer.bias[i] + (1 - beta) * aligned_target_bias[i]
                if source_layer.bias.size(0) > min_bias_dim:
                    fused_bias = torch.cat([fused_bias, source_layer.bias[min_bias_dim:]], dim=0)
            return fused_weight, fused_bias
        except Exception:
            return source_layer.weight.data.clone(), source_layer.bias.data.clone() if source_layer.bias is not None else None


def choose_statistical_method(layer_name, params_shape, base_method="repair"):
    """根據層類型和參數形狀選擇統計對齊方法"""
    if "convm2_layer" in layer_name.lower():
        if len(params_shape) == 4:
            return "repair"
        elif len(params_shape) == 1:
            if "bias" in layer_name and "bn" not in layer_name:
                return "rescale"
            elif "bn" in layer_name and "bias" in layer_name:
                return "rescale"
            elif "bn" in layer_name and "weight" in layer_name:
                return "repair"
            return "repair"
        return "repair"
    elif "conv" in layer_name.lower() or len(params_shape) == 4:
        return "repair"
    elif "bn" in layer_name.lower() or "batch" in layer_name.lower():
        return "rescale"
    elif "linear" in layer_name.lower() or "fc" in layer_name.lower() or len(params_shape) == 2:
        return "repair" if params_shape[0] > 512 else "rescale"
    elif len(params_shape) == 1:
        return "rescale"
    return base_method


def compute_channel_stats(tensor):
    """計算每個通道的統計特徵 [out_channels, 4]"""
    if len(tensor.shape) == 4:
        mean_vals = torch.mean(tensor, dim=(1, 2, 3))
        std_vals = torch.std(tensor, dim=(1, 2, 3))
        max_vals = torch.max(torch.max(torch.max(tensor, dim=3)[0], dim=2)[0], dim=1)[0]
        min_vals = torch.min(torch.min(torch.min(tensor, dim=3)[0], dim=2)[0], dim=1)[0]
    elif len(tensor.shape) == 2:
        mean_vals = torch.mean(tensor, dim=1)
        std_vals = torch.std(tensor, dim=1)
        max_vals = torch.max(tensor, dim=1)[0]
        min_vals = torch.min(tensor, dim=1)[0]
    else:
        raise ValueError(f"不支援的張量維度: {tensor.shape}")
    return torch.stack([mean_vals, std_vals, max_vals, min_vals], dim=1)


def compute_similarity_matrix(stats_0, stats_1):
    """計算兩組統計特徵之間的餘弦相似度矩陣"""
    stats_0_norm = F.normalize(stats_0, p=2, dim=1)
    stats_1_norm = F.normalize(stats_1, p=2, dim=1)
    return torch.mm(stats_0_norm, stats_1_norm.t())


def lightweight_channel_similarity_alignment(params_0, params_1, top_k_ratio=0.3):
    """輕量化通道相似性對齊，返回 (aligned_params_1, channel_mapping)"""
    stats_0 = compute_channel_stats(params_0)
    stats_1 = compute_channel_stats(params_1)
    similarity_matrix = compute_similarity_matrix(stats_0, stats_1)
    num_channels = params_0.shape[0]
    num_realign = int(num_channels * top_k_ratio)
    if num_realign == 0:
        return params_1, list(range(num_channels))

    similarity_flat = similarity_matrix.view(-1)
    _, top_indices = torch.topk(similarity_flat, num_realign)
    row_indices = top_indices // num_channels
    col_indices = top_indices % num_channels
    cost_matrix = (1 - similarity_matrix).cpu().numpy()

    if num_realign < num_channels:
        selected_rows = row_indices.cpu().numpy()
        selected_cols = col_indices.cpu().numpy()
        unique_rows, unique_cols, seen_rows, seen_cols = [], [], set(), set()
        for r, c in zip(selected_rows, selected_cols):
            if r not in seen_rows and c not in seen_cols:
                unique_rows.append(r)
                unique_cols.append(c)
                seen_rows.add(r)
                seen_cols.add(c)
                if len(unique_rows) >= num_realign:
                    break
        if len(unique_rows) > 0:
            sub_cost_matrix = cost_matrix[np.ix_(unique_rows, unique_cols)]
            sub_row_ind, sub_col_ind = linear_sum_assignment(sub_cost_matrix)
            optimal_rows = [unique_rows[i] for i in sub_row_ind]
            optimal_cols = [unique_cols[i] for i in sub_col_ind]
        else:
            optimal_rows, optimal_cols = [], []
    else:
        optimal_rows, optimal_cols = linear_sum_assignment(cost_matrix)

    channel_mapping = list(range(num_channels))
    for model_0_idx, model_1_idx in zip(optimal_rows, optimal_cols):
        channel_mapping[model_0_idx] = model_1_idx

    aligned_params_1 = params_1.clone()
    for target_idx, source_idx in enumerate(channel_mapping):
        if source_idx != target_idx:
            aligned_params_1[target_idx] = params_1[source_idx]
    return aligned_params_1, channel_mapping


def statistical_alignment_fusion(
    params_0,
    params_1,
    alpha=0.5,
    eps=1e-5,
    repair_type="repair",
    layer_name="",
    enable_channel_similarity=False,
    similarity_top_k=0.3,
):
    """統計對齊融合。repair_type: 'repair' | 'rescale' | 'original'"""
    if enable_channel_similarity and params_0.shape == params_1.shape and len(params_0.shape) >= 2:
        try:
            aligned_params_1, _ = lightweight_channel_similarity_alignment(params_0, params_1, top_k_ratio=similarity_top_k)
            params_1 = aligned_params_1
        except Exception as e:
            print(f"  通道相似性對齊失敗: {e}")

    if repair_type == "original":
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
        return alpha * aligned_0 + (1 - alpha) * aligned_1

    if repair_type == "repair":
        if len(params_0.shape) >= 2:
            mean_0 = torch.mean(params_0, dim=(1, 2, 3), keepdim=True) if params_0.dim() == 4 else torch.mean(params_0, dim=1, keepdim=True)
            std_0 = torch.std(params_0, dim=(1, 2, 3), keepdim=True) if params_0.dim() == 4 else torch.std(params_0, dim=1, keepdim=True)
            mean_1 = torch.mean(params_1, dim=(1, 2, 3), keepdim=True) if params_1.dim() == 4 else torch.mean(params_1, dim=1, keepdim=True)
            std_1 = torch.std(params_1, dim=(1, 2, 3), keepdim=True) if params_1.dim() == 4 else torch.std(params_1, dim=1, keepdim=True)
        else:
            mean_0, std_0 = torch.mean(params_0), torch.std(params_0)
            mean_1, std_1 = torch.mean(params_1), torch.std(params_1)
        mean_target = alpha * mean_0 + (1 - alpha) * mean_1
        std_target = alpha * std_0 + (1 - alpha) * std_1
        normalized_0 = (params_0 - mean_0) / (std_0 + eps)
        normalized_1 = (params_1 - mean_1) / (std_1 + eps)
        aligned_0 = normalized_0 * std_target + mean_target
        aligned_1 = normalized_1 * std_target + mean_target
        return alpha * aligned_0 + (1 - alpha) * aligned_1

    if repair_type == "rescale":
        if len(params_0.shape) >= 2:
            if len(params_0.shape) == 4:
                std_0 = torch.std(params_0, dim=(1, 2, 3), keepdim=True)
                std_1 = torch.std(params_1, dim=(1, 2, 3), keepdim=True)
            else:
                std_0 = torch.std(params_0, dim=1, keepdim=True)
                std_1 = torch.std(params_1, dim=1, keepdim=True)
        else:
            std_0, std_1 = torch.std(params_0), torch.std(params_1)
        std_target = alpha * std_0 + (1 - alpha) * std_1
        rescaled_0 = params_0 * std_target / (std_0 + eps)
        rescaled_1 = params_1 * std_target / (std_1 + eps)
        return alpha * rescaled_0 + (1 - alpha) * rescaled_1

    raise ValueError(f"不支援的 repair_type: {repair_type}")
