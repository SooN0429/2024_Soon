#!/bin/bash

# 定義水果列表
fruits=("apple" "banana" "carambola" "guava" "muskmelon" "peach" "pear" "tomato")

# 定義模型列表
models=(
    "models1"
    "models2"
    "models3"
    "models1_1"
    "models1_2"
    "models2_1"
    "models2_2"
    "models3_1"
    "models3_2"
    "models4_1"
    "models4_2"
)

# 基礎日誌目錄
base_log_dir="/mnt/F/Vincent_v1.0/Model_Fusion_LOGs/Model_Fusion_WA_log"

# 解析命令行參數
start_model=""
start_fruit1=""
start_fruit2=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --start-from)
            start_model="$2"
            start_fruit1="$3"
            start_fruit2="$4"
            shift 4
            ;;
        *)
            echo "未知參數: $1"
            exit 1
            ;;
    esac
done

# 找到起始模型的索引
start_model_index=-1
if [[ -n "$start_model" ]]; then
    for i in "${!models[@]}"; do
        if [[ "${models[$i]}" == "$start_model" ]]; then
            start_model_index=$i
            break
        fi
    done
fi

# 外層循環：遍歷不同的模型架構
for current_model_index in "${!models[@]}"; do
    model_name="${models[$current_model_index]}"
    
    # 如果指定了起始模型且還沒到達該模型，則跳過
    if [[ $start_model_index -ne -1 && $current_model_index -lt $start_model_index ]]; then
        continue
    fi
    
    # 從模型名稱中提取數字（去掉"models"）
    model_num=${model_name#models}
    
    # 創建當前模型的實驗名稱和日誌目錄
    #experiment_name="WA_${model_name}<-models_gray<-ori_1"
    experiment_name="WA_${model_name}<-models_color<-ori_1"
    log_dir="${base_log_dir}/${model_name}"
    results_excel="${log_dir}/results_2023_${experiment_name}.xlsx"
    failed_log="${log_dir}/failed_experiments_${experiment_name}.txt"
    
    # 創建日誌目錄
    mkdir -p "$log_dir"
    
    echo "Starting experiments for model architecture: $model_name"
    echo "=================================================="

    # 標記是否已經找到起始組合
    found_start_combination=false
    if [[ $start_model_index -eq -1 || $current_model_index -gt $start_model_index ]]; then
        found_start_combination=true
    elif [[ $current_model_index -eq $start_model_index ]]; then
        # 如果是指定的模型，保持 found_start_combination 為 false，
        # 直到找到指定的水果組合
        found_start_combination=false
    else
        # 如果不是指定的模型，跳過
        continue
    fi

    # 第二層循環：遍歷source_model_1的水果
    for fruit1 in "${fruits[@]}"; do
        # 如果有指定起始組合，檢查 fruit1
        if [[ "$found_start_combination" == "false" && "$fruit1" != "$start_fruit1" ]]; then
            continue
        fi
        
        # 第三層循環：遍歷source_model_2的水果
        for fruit2 in "${fruits[@]}"; do
            # 如果還沒找到起始組合，檢查是否是指定的組合
            if [[ "$found_start_combination" == "false" ]]; then
                if [[ "$fruit1" == "$start_fruit1" && "$fruit2" == "$start_fruit2" ]]; then
                    found_start_combination=true
                else
                    continue
                fi
            fi

            echo "Running experiment for combination: $fruit2 -> $fruit1 using $model_name"
            
            # 修改Python文件中的模型引用 (models<-modelsx_x)
            #sed -i "s/model_0 = models[0-9_]*.Transfer_Net/model_0 = models.Transfer_Net/g" /mnt/F/Vincent/CL_MAL/model_fusion_new_6_4.py
            #sed -i "s/model_1 = models[0-9_]*.Transfer_Net/model_1 = ${model_name}.Transfer_Net/g" /mnt/F/Vincent/CL_MAL/model_fusion_new_6_4.py
            #sed -i "s/model = models[0-9_]*.Transfer_Net/model = models.Transfer_Net/g" /mnt/F/Vincent/CL_MAL/model_fusion_new_6_4.py

            # 修改Python文件中的模型引用 (modelsx_x<-models)
            sed -i "s/model_0 = models[0-9_]*.Transfer_Net/model_0 = ${model_name}.Transfer_Net/g" /mnt/F/Vincent_v1.0/CL_MAL/model_fusion_2023.py
            sed -i "s/model = models[0-9_]*.Transfer_Net/model = ${model_name}.Transfer_Net/g" /mnt/F/Vincent_v1.0/CL_MAL/model_fusion_2023.py           

            # 執行實驗命令
            # save_parameter_path_name left_0不要動，要動要去看學姊的說明檔：model_fusion_2023.py 634行
            pipenv run python /mnt/F/Vincent_v1.0/CL_MAL/model_fusion_2023.py \
                --source_model_1 "$fruit1" \
                --source_train_feature source_train_ColorJitter_feature.npy \
                --source_train_feature_label source_train_ColorJitter_feature_label.npy \
                --source_model_2 "$fruit2" \
                --extracted_layer 8_point \
                --train_root_path /mnt/F/Vincent_v1.0/create_datasets/feature_extract/dataset/feature_map/latent/8/ \
                --test_root_path /mnt/F/Vincent_v1.0/create_datasets/feature_extract/dataset/ColorJitter_test_image/ \
                --source_dir "${fruit1}_to_/to_${fruit1}/source/train/" \
                --target_test_dir "${fruit1}_to_/to_${fruit1}/sn_200_sp_200/target/test/" \
                --batch_size 8 \
                --lr 0.0001 \
                --epoch 50 \
                --save_parameter_path_name "left_0_${model_name}<-models_color<-ori_1" \
                --save_parameter_path_name_1 "epoch50_color_0_model${model_num}.pkl" \
                --save_parameter_path_name_2 "epoch150_0.pkl" \
                --gpu_id cuda:0 \
                --results_excel "$results_excel" \
                --log_dir "$log_dir"
            
            # 檢查命令是否成功執行
            if [ $? -eq 0 ]; then
                echo "Experiment for $fruit2 -> $fruit1 using $model_name completed successfully."
            else
                echo "Experiment for $fruit2 -> $fruit1 using $model_name failed."
                echo "Failed combination: $fruit2 -> $fruit1 ($model_name)" >> "$failed_log"
                continue
            fi
            
            # 每次實驗後暫停1秒
            echo "Cooling down for 1 seconds..."
            sleep 1
        done
    done

    echo "Completed all experiments for $model_name"
    echo "=================================================="
done

# 計算並顯示總體實驗統計信息
total_combinations=$((${#models[@]} * ${#fruits[@]} * ${#fruits[@]}))
echo "All experiments completed."
echo "Total model architectures processed: ${#models[@]}"
echo "Total fruit combinations per model: $((${#fruits[@]} * ${#fruits[@]}))"
echo "Total combinations processed: $total_combinations"
echo "Check individual failed_log files in each model directory for any failed experiments." 