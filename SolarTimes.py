import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
import math
import pickle

plt.rcParams['font.sans-serif'] = ['SimHei']
warnings.filterwarnings('ignore')
# 设置字体 - 添加Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
warnings.filterwarnings('ignore')


# 设置随机种子以确保结果可重现
def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


seed_everything()

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


# 添加TimeMixerConfig类
class TimeMixerConfig:
    def __init__(self):
        # TimeMixer特有参数
        self.decomp_method = 'moving_avg'  # 分解方法
# 1. 加载数据
def load_data(file_path):
    """加载数据并进行基本处理"""
    try:
        # 尝试自动检测分隔符
        df = pd.read_csv(file_path, engine='python',encoding='gb2312')
        print(df.head())

        # 检查数据是否加载正确
        if len(df.columns) < 5:  # 假设至少应该有5列
            print(f"警告：数据列数少于预期 ({len(df.columns)}列)")

        # 转换日期列
        if '日期' in df.columns:
            df['date'] = pd.to_datetime(df['日期'])
        else:
            print("警告：未找到'日期'列，尝试查找其他可能的日期列")
            date_cols = [col for col in df.columns if 'date' in col.lower() or '日期' in col]
            if date_cols:
                df['date'] = pd.to_datetime(df[date_cols[0]])
            else:
                print("无法找到日期列，创建虚拟日期")
                df['date'] = pd.date_range(start='2024-01-01', periods=len(df))

        # 提取功率列和气象列
        power_cols = [col for col in df.columns if col.startswith('p') and not col.startswith('w_p')]
        weather_cols = [col for col in df.columns if col.startswith('w_p')]

        print(f"找到 {len(power_cols)} 个功率列和 {len(weather_cols)} 个气象列")

        # 检查机组类型分布
        if 'unit_type' in df.columns:
            unit_counts = df['unit_type'].value_counts()
            print(f"机组类型分布:\n{unit_counts}")
        else:
            print("警告：未找到'unit_type'列")

        return df, power_cols, weather_cols

    except Exception as e:
        print(f"加载数据时出错: {e}")
        raise

class TimeSeriesDatasetWithDualTargets(Dataset):
    def __init__(self, features, power_targets, ssrd_targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.power_targets = torch.tensor(power_targets, dtype=torch.float32)
        self.ssrd_targets = torch.tensor(ssrd_targets, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.power_targets[idx], self.ssrd_targets[idx]
# 2. 加载机组容量限制数据
def load_unit_capacity_limits(file_path='MinMax.xlsx'):
    """
    读取机组容量限制数据

    Args:
        file_path: MinMax.xlsx文件路径

    Returns:
        unit_caps: 包含各机组容量限制的字典 {unit_id: (min_cap, max_cap)}
    """
    try:
        # 读取Excel文件
        df = pd.read_excel(file_path)

        # 确保必需的列存在
        required_cols = ['unit_id', 'MAX_CAP', 'MIN_CAP']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"MinMax.xlsx应包含以下列: {required_cols}，但实际列为: {df.columns}")

        # 创建机组容量限制字典
        unit_caps = {}
        for _, row in df.iterrows():
            unit_id = int(row['unit_id'])  # 确保unit_id为整数
            min_cap = float(row['MIN_CAP'])
            max_cap = float(row['MAX_CAP'])
            unit_caps[unit_id] = (min_cap, max_cap)

        print(f"已加载{len(unit_caps)}个机组的容量限制数据")
        return unit_caps

    except Exception as e:
        print(f"加载MinMax.xlsx时出错: {e}")
        # 返回空字典以便程序继续运行
        return {}


# 3. 准备数据，添加标准化处理


def prepare_time_series_data(df, power_cols, weather_cols, input_days, forecast_days, test_split=0.2, slide_by='step'):
    """将数据预处理修改为同时准备功率和SSRD目标"""
    from collections import defaultdict

    # 先过滤只保留光伏
    df_solar = df[df['unit_type'] == 'solar'].copy()

    print(f"Original dataset size: {len(df)} rows")
    print(f"Solar-only dataset size: {len(df_solar)} rows")
    print(f"Filtered out {len(df) - len(df_solar)} rows of wind power data")

    # 检查有多少唯一的光伏机组
    solar_units = df_solar['unit_name'].unique()
    print(f"Number of unique solar units: {len(solar_units)}")

    # 后续代码与原来类似
    steps_per_day = len(power_cols)  # 96
    input_seq_len = int(input_days * steps_per_day)
    output_seq_len = int(forecast_days * steps_per_day)

    unit_type_encoder = {'solar': [0, 1]}

    all_units = solar_units  # 只使用光伏机组
    df_solar['date'] = pd.to_datetime(df_solar['date'])
    all_dates = sorted(df_solar['date'].unique())
    split_idx = int(len(all_dates) * (1 - test_split))
    train_dates = all_dates[:split_idx]
    test_dates = all_dates[split_idx:]

    train_df = df_solar[df_solar['date'].isin(train_dates)]
    test_df = df_solar[df_solar['date'].isin(test_dates)]

    # 创建标准化器字典
    scalers = {
        'power': StandardScaler(),  # 功率数据标准化器
        'weather': StandardScaler(),  # 气象数据标准化器
        'target': StandardScaler()  # 目标数据标准化器
    }

    # 收集所有特征数据来拟合标准化器
    all_power_data = []
    all_weather_data = []
    all_target_data = []

    # 仅从训练集拟合标准化器
    for unit in all_units:
        unit_train_df = train_df[train_df['unit_name'] == unit]
        if len(unit_train_df) < input_days + forecast_days:
            continue

        # 收集数据用于拟合标准化器
        unit_train_df = unit_train_df.sort_values(['date'])
        power_data = unit_train_df[power_cols].values
        weather_data = unit_train_df[weather_cols].values

        all_power_data.append(power_data.flatten())
        all_weather_data.append(weather_data.flatten())
        all_target_data.append(power_data.flatten())  # 目标也是功率数据

    # 拟合标准化器
    if all_power_data:
        all_power_data = np.concatenate(all_power_data).reshape(-1, 1)
        all_weather_data = np.concatenate(all_weather_data).reshape(-1, 1)
        all_target_data = np.concatenate(all_target_data).reshape(-1, 1)

        print(f"Fitting scalers, power data shape: {all_power_data.shape}")
        print(f"Fitting scalers, weather data shape: {all_weather_data.shape}")

        scalers['power'].fit(all_power_data)
        scalers['weather'].fit(all_weather_data)
        scalers['target'].fit(all_target_data)

        # 保存标准化器以便后续使用
        with open('scalers_solar_only.pkl', 'wb') as f:
            pickle.dump(scalers, f)

        print("标准化器已拟合并保存")
    else:
        print("Warning: Not enough data to fit scalers")

    def process_unit(unit_df, is_train):
        unit_df = unit_df.sort_values(['date'])

        # 获取原始数据
        power_data = unit_df[power_cols].values
        weather_data = unit_df[weather_cols].values

        # 标准化数据
        power_data_normalized = scalers['power'].transform(power_data.flatten().reshape(-1, 1)).reshape(
            power_data.shape)
        weather_data_normalized = scalers['weather'].transform(weather_data.flatten().reshape(-1, 1)).reshape(
            weather_data.shape)

        # 展平成 1D 序列
        power_seq = power_data_normalized.flatten()  # shape: [total_steps]
        weather_seq = weather_data_normalized.flatten()  # shape: [total_steps]
        time_seq = np.array([(i % steps_per_day) / steps_per_day for i in range(len(power_seq))])

        # 机组类型
        unit_type = unit_df['unit_type'].iloc[0]
        unit_onehot = unit_type_encoder[unit_type]  # [0, 1]

        # 机组ID
        unit_id = unit_df['unit_name'].iloc[0]

        features, power_targets, ssrd_targets, types, target_indices, unit_ids, raw_power_targets, raw_ssrd_targets = [], [], [], [], [], [], [], []

        # 滑动窗口步长设置
        if isinstance(slide_by, str):
            step = steps_per_day if slide_by == 'day' else 1
        else:
            step = int(slide_by)

        total_len = len(power_seq)
        for i in range(0, total_len - input_seq_len - output_seq_len + 1, step):
            # 输入特征
            X_power = power_seq[i:i + input_seq_len]
            X_weather = weather_seq[i:i + input_seq_len]
            X_time = time_seq[i:i + input_seq_len]

            # 目标数据 - 功率 (保存标准化和原始)
            y_power_raw = power_data.flatten()[i + input_seq_len:i + input_seq_len + output_seq_len]
            y_power = power_seq[i + input_seq_len:i + input_seq_len + output_seq_len]

            # 目标数据 - SSRD (保存标准化和原始)
            y_ssrd_raw = weather_data.flatten()[i + input_seq_len:i + input_seq_len + output_seq_len]
            y_ssrd = weather_seq[i + input_seq_len:i + input_seq_len + output_seq_len]

            # 拼接特征
            X = np.concatenate([
                X_power,  # input_seq_len
                X_weather,  # input_seq_len
                X_time,  # input_seq_len
                unit_onehot,  # 2维
                np.array([unit_id])  # 1维
            ])

            features.append(X)
            power_targets.append(y_power)
            ssrd_targets.append(y_ssrd)  # 添加SSRD目标
            raw_power_targets.append(y_power_raw)
            raw_ssrd_targets.append(y_ssrd_raw)  # 添加原始SSRD目标
            types.append(unit_type)
            target_indices.append((i + input_seq_len, i + input_seq_len + output_seq_len))
            unit_ids.append(unit_id)

        return features, power_targets, ssrd_targets, types, target_indices, unit_ids, raw_power_targets, raw_ssrd_targets

    train_features, train_power_targets, train_ssrd_targets = [], [], []
    train_types, train_indices, train_unit_ids = [], [], []
    train_raw_power_targets, train_raw_ssrd_targets = [], []

    test_features, test_power_targets, test_ssrd_targets = [], [], []
    test_types, test_indices, test_unit_ids = [], [], []
    test_raw_power_targets, test_raw_ssrd_targets = [], []

    for unit in tqdm(all_units, desc="Processing unit data"):
        unit_train_df = train_df[train_df['unit_name'] == unit]
        unit_test_df = test_df[test_df['unit_name'] == unit]

        if len(unit_train_df) < input_days + forecast_days or len(unit_test_df) < input_days + forecast_days:
            continue

        feats, p_tgts, s_tgts, types, indices, unit_ids, raw_p_tgts, raw_s_tgts = process_unit(unit_train_df,
                                                                                               is_train=True)
        train_features.extend(feats)
        train_power_targets.extend(p_tgts)
        train_ssrd_targets.extend(s_tgts)
        train_types.extend(types)
        train_indices.extend(indices)
        train_unit_ids.extend(unit_ids)
        train_raw_power_targets.extend(raw_p_tgts)
        train_raw_ssrd_targets.extend(raw_s_tgts)

        feats, p_tgts, s_tgts, types, indices, unit_ids, raw_p_tgts, raw_s_tgts = process_unit(unit_test_df,
                                                                                               is_train=False)
        test_features.extend(feats)
        test_power_targets.extend(p_tgts)
        test_ssrd_targets.extend(s_tgts)
        test_types.extend(types)
        test_indices.extend(indices)
        test_unit_ids.extend(unit_ids)
        test_raw_power_targets.extend(raw_p_tgts)
        test_raw_ssrd_targets.extend(raw_s_tgts)

    # 创建数据集 - 现在需要同时传递功率和SSRD目标
    train_dataset = TimeSeriesDatasetWithDualTargets(
        np.array(train_features),
        np.array(train_power_targets),
        np.array(train_ssrd_targets)
    )

    test_dataset = TimeSeriesDatasetWithDualTargets(
        np.array(test_features),
        np.array(test_power_targets),
        np.array(test_ssrd_targets)
    )

    print(f"Created train dataset with {len(train_dataset)} samples")
    print(f"Created test dataset with {len(test_dataset)} samples")

    return {
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        'power_cols': power_cols,
        'weather_cols': weather_cols,
        'steps_per_day': steps_per_day,
        'input_seq_len': input_seq_len,
        'output_seq_len': output_seq_len,
        'train_unit_types': np.array(train_types),
        'test_unit_types': np.array(test_types),
        'train_unit_ids': np.array(train_unit_ids),
        'test_unit_ids': np.array(test_unit_ids),
        'train_raw_power_targets': np.array(train_raw_power_targets),
        'test_raw_power_targets': np.array(test_raw_power_targets),
        'train_raw_ssrd_targets': np.array(train_raw_ssrd_targets),  # 添加SSRD原始目标
        'test_raw_ssrd_targets': np.array(test_raw_ssrd_targets),  # 添加SSRD原始目标
        'scalers': scalers,
        'train_indices': train_indices,
        'test_indices': test_indices
    }


# 4. 定义时间序列数据集类
# class TimeSeriesDataset(Dataset):
#     def __init__(self, features, targets):
#         self.features = torch.tensor(features, dtype=torch.float32)
#         self.targets = torch.tensor(targets, dtype=torch.float32)
#
#     def __len__(self):
#         return len(self.features)
#
#     def __getitem__(self, idx):
#         return self.features[idx], self.targets[idx]


# 5. 创建训练、验证和测试集的数据加载器

# 修改数据加载器创建函数以支持双目标
def create_train_val_test_loaders(data_dict, batch_size=32, val_split=0.2):
    """
    创建训练、验证和测试数据加载器，支持双目标（功率和SSRD）
    """
    # 获取完整训练集
    train_dataset = data_dict['train_dataset']

    # 计算训练集和验证集的大小
    train_size = int(len(train_dataset) * (1 - val_split))
    val_size = len(train_dataset) - train_size

    # 随机分割训练集和验证集
    indices = torch.randperm(len(train_dataset)).tolist()
    train_subset = torch.utils.data.Subset(train_dataset, indices[:train_size])
    val_subset = torch.utils.data.Subset(train_dataset, indices[train_size:])

    # 创建数据加载器
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )

    test_loader = DataLoader(
        data_dict['test_dataset'],
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader



def train_model(model, train_loader, val_loader, optimizer, power_criterion, ssrd_criterion,
                num_epochs, device, ssrd_weight=0.3, scheduler=None):
    """
    """
    # 将模型移动到指定设备
    model = model.to(device)

    # 记录训练历史
    history = {
        'train_loss': [],
        'train_power_loss': [],
        'train_ssrd_loss': [],
        'val_loss': [],
        'val_power_loss': [],
        'val_ssrd_loss': []
    }

    best_val_loss = float('inf')
    patience = 15  # 控制早停
    patience_counter = 0

    # 训练循环
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_power_loss = 0.0
        train_ssrd_loss = 0.0
        train_steps = 0

        for inputs, power_targets, ssrd_targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]"):
            # 将数据移动到设备
            inputs = inputs.to(device)
            power_targets = power_targets.to(device)
            ssrd_targets = ssrd_targets.to(device)  # 使用真实SSRD目标

            # 前向传播
            power_outputs, ssrd_outputs = model(inputs)

            # 计算功率预测损失
            power_loss = power_criterion(power_outputs, power_targets)

            # 计算SSRD预测损失 - 现在使用真实SSRD目标
            ssrd_loss = ssrd_criterion(ssrd_outputs, ssrd_targets)

            # 组合损失 (1-ssrd_weight为功率权重)
            power_weight = 1.0 - ssrd_weight
            combined_loss = power_weight * power_loss + ssrd_weight * ssrd_loss

            # 反向传播和优化
            optimizer.zero_grad()
            combined_loss.backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += combined_loss.item()
            train_power_loss += power_loss.item()
            train_ssrd_loss += ssrd_loss.item()
            train_steps += 1

        # 计算平均训练损失
        train_loss /= train_steps
        train_power_loss /= train_steps
        train_ssrd_loss /= train_steps
        history['train_loss'].append(train_loss)
        history['train_power_loss'].append(train_power_loss)
        history['train_ssrd_loss'].append(train_ssrd_loss)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_power_loss = 0.0
        val_ssrd_loss = 0.0
        val_steps = 0

        with torch.no_grad():
            for inputs, power_targets, ssrd_targets in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]"):
                # 将数据移动到设备
                inputs = inputs.to(device)
                power_targets = power_targets.to(device)
                ssrd_targets = ssrd_targets.to(device)  # 使用真实SSRD目标

                # 前向传播
                power_outputs, ssrd_outputs = model(inputs)

                # 计算损失
                power_loss = power_criterion(power_outputs, power_targets)
                ssrd_loss = ssrd_criterion(ssrd_outputs, ssrd_targets)  # 使用真实SSRD目标

                # 组合损失
                power_weight = 1.0 - ssrd_weight
                combined_loss = power_weight * power_loss + ssrd_weight * ssrd_loss

                val_loss += combined_loss.item()
                val_power_loss += power_loss.item()
                val_ssrd_loss += ssrd_loss.item()
                val_steps += 1

        # 计算平均验证损失
        val_loss /= val_steps
        val_power_loss /= val_steps
        val_ssrd_loss /= val_steps
        history['val_loss'].append(val_loss)
        history['val_power_loss'].append(val_power_loss)
        history['val_ssrd_loss'].append(val_ssrd_loss)

        # 打印当前epoch的损失
        print(
            f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f} (Power: {train_power_loss:.4f}, SSRD: {train_ssrd_loss:.4f}), Val Loss: {val_loss:.4f} (Power: {val_power_loss:.4f}, SSRD: {val_ssrd_loss:.4f})")

        # 学习率调度器
        if scheduler is not None:
            scheduler.step(val_loss)

        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Saved best model with validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

    return history


# 修改评估函数以使用真实SSRD目标
def evaluate_model_performance(model, test_loader, power_criterion, ssrd_criterion, device, data_dict,
                               dataset_type='test', ssrd_weight=0.3):
    """
    评估模型，支持双输出模型，使用真实SSRD目标
    """
    model = model.to(device)
    model.eval()

    test_loss = 0.0
    test_power_loss = 0.0
    test_ssrd_loss = 0.0

    all_power_preds = []
    all_ssrd_preds = []
    all_power_targets = []
    all_ssrd_targets = []
    all_ids = []  # 收集机组ID

    # 获取机组ID列表和原始目标数据
    unit_ids = data_dict[f'{dataset_type}_unit_ids']
    raw_power_targets = data_dict[f'{dataset_type}_raw_power_targets']
    raw_ssrd_targets = data_dict[f'{dataset_type}_raw_ssrd_targets']

    # 获取索引信息，包含时间信息
    time_indices = data_dict[f'{dataset_type}_indices']

    # 每天的时间步数
    steps_per_day = data_dict['steps_per_day']

    # 跟踪当前样本索引
    current_idx = 0

    with torch.no_grad():
        for inputs, power_targets, ssrd_targets in tqdm(test_loader, desc=f"Evaluating {dataset_type} set"):
            inputs = inputs.to(device)
            power_targets = power_targets.to(device)
            ssrd_targets = ssrd_targets.to(device)  # 使用真实SSRD目标

            # 前向传播
            power_outputs, ssrd_outputs = model(inputs)

            # 计算损失
            power_loss = power_criterion(power_outputs, power_targets)
            ssrd_loss = ssrd_criterion(ssrd_outputs, ssrd_targets)  # 使用真实SSRD目标

            # 组合损失
            power_weight = 1.0 - ssrd_weight
            combined_loss = power_weight * power_loss + ssrd_weight * ssrd_loss

            test_loss += combined_loss.item()
            test_power_loss += power_loss.item()
            test_ssrd_loss += ssrd_loss.item()

            # 收集预测和目标（标准化尺度）
            current_power_preds = power_outputs.cpu().numpy()
            current_ssrd_preds = ssrd_outputs.cpu().numpy()
            current_power_targets = power_targets.cpu().numpy()
            current_ssrd_targets = ssrd_targets.cpu().numpy()

            # 获取当前批次的机组ID
            batch_size = len(current_power_preds)
            batch_ids = unit_ids[current_idx:current_idx + batch_size]
            current_idx += batch_size

            all_power_preds.append(current_power_preds)
            all_ssrd_preds.append(current_ssrd_preds)
            all_power_targets.append(current_power_targets)
            all_ssrd_targets.append(current_ssrd_targets)
            all_ids.extend(batch_ids)

    # 计算平均测试损失
    test_loss /= len(test_loader)
    test_power_loss /= len(test_loader)
    test_ssrd_loss /= len(test_loader)

    # 合并所有预测和目标
    all_power_preds = np.concatenate(all_power_preds, axis=0)
    all_ssrd_preds = np.concatenate(all_ssrd_preds, axis=0)
    all_power_targets = np.concatenate(all_power_targets, axis=0)
    all_ssrd_targets = np.concatenate(all_ssrd_targets, axis=0)

    # 计算标准化尺度的误差 - 功率
    power_normalized_mae = np.mean(np.abs(all_power_preds - all_power_targets))
    power_normalized_rmse = np.sqrt(np.mean((all_power_preds - all_power_targets) ** 2))
    power_normalized_r2 = r2_score(all_power_targets.flatten(), all_power_preds.flatten())

    # 计算标准化尺度的误差 - SSRD
    ssrd_normalized_mae = np.mean(np.abs(all_ssrd_preds - all_ssrd_targets))
    ssrd_normalized_rmse = np.sqrt(np.mean((all_ssrd_preds - all_ssrd_targets) ** 2))
    ssrd_normalized_r2 = r2_score(all_ssrd_targets.flatten(), all_ssrd_preds.flatten())

    # 将预测从标准化尺度转换回原始尺度 - 功率
    power_scaler = data_dict['scalers']['target']
    weather_scaler = data_dict['scalers']['weather']

    all_power_preds_original = power_scaler.inverse_transform(all_power_preds.reshape(-1, 1)).reshape(
        all_power_preds.shape)
    all_power_preds_original = np.maximum(all_power_preds_original, 0)

    # 将SSRD预测转换为原始尺度
    all_ssrd_preds_original = weather_scaler.inverse_transform(all_ssrd_preds.reshape(-1, 1)).reshape(
        all_ssrd_preds.shape)
    all_ssrd_preds_original = np.maximum(all_ssrd_preds_original, 0)  # SSRD不应为负

    # 使用原始未标准化的目标值
    all_power_targets_original = raw_power_targets
    all_ssrd_targets_original = raw_ssrd_targets

    # 计算原始尺度的误差 - 功率
    power_original_mae = np.mean(np.abs(all_power_preds_original - all_power_targets_original))
    power_original_rmse = np.sqrt(np.mean((all_power_preds_original - all_power_targets_original) ** 2))
    power_original_r2 = r2_score(all_power_targets_original.flatten(), all_power_preds_original.flatten())

    # 计算原始尺度的误差 - SSRD
    ssrd_original_mae = np.mean(np.abs(all_ssrd_preds_original - all_ssrd_targets_original))
    ssrd_original_rmse = np.sqrt(np.mean((all_ssrd_preds_original - all_ssrd_targets_original) ** 2))
    ssrd_original_r2 = r2_score(all_ssrd_targets_original.flatten(), all_ssrd_preds_original.flatten())

    # 打印评估结果
    print(f"{dataset_type} 损失 (总体): {test_loss:.4f}")
    print(f"{dataset_type} 损失 (功率): {test_power_loss:.4f}")
    print(f"{dataset_type} 损失 (SSRD): {test_ssrd_loss:.4f}")
    print(
        f"{dataset_type} 功率 - 标准化尺度 - MAE: {power_normalized_mae:.4f}, RMSE: {power_normalized_rmse:.4f}, R²: {power_normalized_r2:.4f}")
    print(
        f"{dataset_type} 功率 - 原始尺度 - MAE: {power_original_mae:.4f}, RMSE: {power_original_rmse:.4f}, R²: {power_original_r2:.4f}")
    print(
        f"{dataset_type} SSRD - 标准化尺度 - MAE: {ssrd_normalized_mae:.4f}, RMSE: {ssrd_normalized_rmse:.4f}, R²: {ssrd_normalized_r2:.4f}")
    print(
        f"{dataset_type} SSRD - 原始尺度 - MAE: {ssrd_original_mae:.4f}, RMSE: {ssrd_original_rmse:.4f}, R²: {ssrd_original_r2:.4f}")

    # 保存预测结果为CSV
    results_df_data = []

    # 输出序列长度
    output_seq_len = data_dict['output_seq_len']

    # 为每个样本和时间步创建一行
    for i in range(len(all_power_preds_original)):
        unit_id = all_ids[i]

        # 获取样本的起始索引
        start_idx, end_idx = time_indices[i]

        for j in range(output_seq_len):
            # 计算当前时间步的绝对索引
            abs_time_idx = start_idx + j

            # 计算时间点信息: 哪一天的哪个时间点
            day_idx = abs_time_idx // steps_per_day
            time_of_day_idx = abs_time_idx % steps_per_day

            results_df_data.append({
                'unit_id': unit_id,
                'day_idx': day_idx,
                'time_step': time_of_day_idx,
                'predicted_power': all_power_preds_original[i, j],
                'actual_power': all_power_targets_original[i, j],
                'predicted_ssrd': all_ssrd_preds_original[i, j],
                'actual_ssrd': all_ssrd_targets_original[i, j]
            })

    # 创建DataFrame并保存为CSV
    results_df = pd.DataFrame(results_df_data)
    csv_filename = f'{dataset_type}_dual_predictions_results.csv'
    results_df.to_csv(csv_filename, index=False)
    print(f"预测结果已保存到 {csv_filename}")

    return {
        'test_loss': test_loss,
        'test_power_loss': test_power_loss,
        'test_ssrd_loss': test_ssrd_loss,
        'power_normalized_mae': power_normalized_mae,
        'power_normalized_rmse': power_normalized_rmse,
        'power_normalized_r2': power_normalized_r2,
        'power_original_mae': power_original_mae,
        'power_original_rmse': power_original_rmse,
        'power_original_r2': power_original_r2,
        'ssrd_normalized_mae': ssrd_normalized_mae,
        'ssrd_normalized_rmse': ssrd_normalized_rmse,
        'ssrd_normalized_r2': ssrd_normalized_r2,
        'ssrd_original_mae': ssrd_original_mae,
        'ssrd_original_rmse': ssrd_original_rmse,
        'ssrd_original_r2': ssrd_original_r2,
        'power_predictions': all_power_preds,
        'power_targets': all_power_targets,
        'ssrd_predictions': all_ssrd_preds,
        'ssrd_targets': all_ssrd_targets,
        'power_predictions_original': all_power_preds_original,
        'power_targets_original': all_power_targets_original,
        'ssrd_predictions_original': all_ssrd_preds_original,
        'ssrd_targets_original': all_ssrd_targets_original,
        'unit_ids': all_ids
    }


class SolarSSRDActivation(nn.Module):
    """
    高效并行版光伏SSRD激活层，使用拉格朗日乘子法约束
    """

    def __init__(self,
                 unit_capacity_dict,
                 c_prime=1.0,
                 alpha=0.3,
                 alpha_prime=0.4,
                 A=1.0,
                 eta=0.2,
                 default_max_power=500,
                 default_min_power=0.0,
                 ssrd_scale=0.0001,
                 learnable=True,
                 balance_constraint=True):
        super(SolarSSRDActivation, self).__init__()
        self.unit_capacity_dict = unit_capacity_dict
        self.default_max_power = default_max_power
        self.default_min_power = default_min_power
        self.balance_constraint = balance_constraint

        if learnable:
            self.c_prime = nn.Parameter(torch.tensor(c_prime, dtype=torch.float))
            self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float))
            self.alpha_prime = nn.Parameter(torch.tensor(alpha_prime, dtype=torch.float))
            self.register_buffer('A', torch.tensor(10.08, dtype=torch.float))
            self.register_buffer('eta', torch.tensor(0.2721, dtype=torch.float))
            self.ssrd_scale = nn.Parameter(torch.tensor(ssrd_scale, dtype=torch.float))
        else:
            self.register_buffer('c_prime', torch.tensor(c_prime, dtype=torch.float))
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float))
            self.register_buffer('alpha_prime', torch.tensor(alpha_prime, dtype=torch.float))
            self.register_buffer('A', torch.tensor(A, dtype=torch.float))
            self.register_buffer('eta', torch.tensor(eta, dtype=torch.float))
            self.register_buffer('ssrd_scale', torch.tensor(ssrd_scale, dtype=torch.float))

    def forward(self, x, weather_data, is_solar, unit_ids):
        result = F.relu(x)
        solar_mask = (is_solar == 1).squeeze(-1)
        if solar_mask.any():
            solar_indices = solar_mask.nonzero().squeeze(-1)
            x_solar = x[solar_indices]

            # Ensure weather_data matches x_solar's seq_len
            if weather_data.size(1) != x_solar.size(1):
                # Truncate or pad weather_data to match seq_len
                target_seq_len = x_solar.size(1)
                if weather_data.size(1) > target_seq_len:
                    weather_data = weather_data[:, :target_seq_len]
                else:
                    weather_data = F.pad(weather_data, (0, target_seq_len - weather_data.size(1)), mode='replicate')

            ssrd = weather_data[solar_indices].unsqueeze(-1)
            solar_unit_ids = unit_ids[solar_indices].int()

            max_powers = []
            min_powers = []
            for unit_id in solar_unit_ids:
                unit_id_int = unit_id.item()
                if unit_id_int in self.unit_capacity_dict:
                    min_cap, max_cap = self.unit_capacity_dict[unit_id_int]
                else:
                    min_cap, max_cap = self.default_min_power, self.default_max_power
                max_powers.append(max_cap)
                min_powers.append(min_cap)

            max_powers = torch.tensor(max_powers, device=x.device).float().unsqueeze(-1).unsqueeze(-1)
            min_powers = torch.tensor(min_powers, device=x.device).float().unsqueeze(-1).unsqueeze(-1)

            solar_result = self._apply_physics_constrained_activation(x_solar, ssrd, max_powers, min_powers)
            new_result = result.clone()
            new_result[solar_indices] = solar_result
            return new_result

        return result

    def _apply_physics_constrained_activation(self, x, ssrd, max_powers, min_powers):
        batch_size, seq_len, dim = x.shape
        ssrd_norm = torch.clamp(ssrd * self.ssrd_scale, min=0.01, max=1.0)
        activation_factor = self.c_prime * (self.A * self.eta * ssrd_norm) / (self.alpha + self.alpha_prime)*max_powers
        x_activated = x * activation_factor

        lower_bounds = min_powers
        upper_bounds = max_powers

        if not self.balance_constraint:
            return torch.clamp(x_activated, min=lower_bounds, max=upper_bounds)

        target_sum = x_activated.sum(dim=-1, keepdim=True)
        lambda_val = self._parallel_optimization(x_activated, lower_bounds, upper_bounds, target_sum)
        result = torch.clamp(x_activated - lambda_val, min=lower_bounds, max=upper_bounds)
        return result

    def _parallel_optimization(self, x, lower, upper, target_sum, max_iter=5, tol=1e-1):
        batch_size, seq_len, dim = x.shape
        x_flat = x.reshape(batch_size * seq_len, dim)
        max_val = x_flat.max(dim=-1)[0].reshape(batch_size, seq_len, 1)
        min_val = x_flat.min(dim=-1)[0].reshape(batch_size, seq_len, 1)
        range_val = torch.clamp(max_val - min_val, min=1.0)

        lambda_min = -range_val
        lambda_max = range_val
        lower = lower.expand_as(x)
        upper = upper.expand_as(x)

        for _ in range(max_iter):
            lambda_mid = (lambda_min + lambda_max) / 2.0
            y_new = torch.clamp(x - lambda_mid, min=lower, max=upper)
            total_new = y_new.sum(dim=-1, keepdim=True)
            diff = total_new - target_sum
            converged = (diff.abs() < tol)
            lambda_min = torch.where((total_new > target_sum) & ~converged, lambda_mid, lambda_min)
            lambda_max = torch.where((total_new <= target_sum) & ~converged, lambda_mid, lambda_max)
            if converged.all():
                break

        return (lambda_min + lambda_max) / 2.0


class SolarModifiedPDM(nn.Module):
    """
    修改版PDM模块，为光伏机组添加SSRD激活，带跳层连接
    """

    def __init__(self, configs, unit_capacity_dict, skip_weight=0.5):
        super(SolarModifiedPDM, self).__init__()
        self.seq_len = configs.seq_len
        self.hidden_dim = configs.d_model
        self.ma = nn.AvgPool1d(kernel_size=25, stride=1, padding=12)
        self.season_proj = nn.Linear(configs.d_model, configs.d_model)
        self.trend_proj = nn.Linear(configs.d_model, configs.d_model)
        self.solar_activation = SolarSSRDActivation(
            unit_capacity_dict=unit_capacity_dict,
            A=10.0,
            eta=0.2,
            learnable=True,
            balance_constraint=True
        )
        self.skip_weight = nn.Parameter(torch.tensor(skip_weight, dtype=torch.float))

    def forward(self, x, weather_data=None, is_solar=None, unit_ids=None):
        batch_size, seq_len, hidden_dim = x.shape
        x_conv = x.transpose(1, 2)
        trend = self.ma(x_conv)
        if trend.shape[2] != seq_len:
            pad_size = seq_len - trend.shape[2]
            trend = F.pad(trend, (0, pad_size), "replicate")
        seasonal = x_conv - trend
        trend = trend.transpose(1, 2)
        seasonal = seasonal.transpose(1, 2)
        seasonal = self.season_proj(seasonal)
        trend = self.trend_proj(trend)
        output = seasonal + trend

        output_raw = output.clone()
        if weather_data is not None and is_solar is not None and unit_ids is not None:
            output_activated = self.solar_activation(output, weather_data, is_solar, unit_ids)
            output = self.skip_weight * output_activated + (1.0 - self.skip_weight) * output_raw
        else:
            output = F.relu(output)

        return output


class SolarModifiedTimeMixerModel(nn.Module):
    """
    修改版TimeMixer++模型，同时预测功率和SSRD，并使用预测的SSRD激活功率输出
    """

    def __init__(self, input_dim, hidden_dim, output_seq_len, input_seq_len, unit_capacity_dict=None, skip_weight=0.9):
        super(SolarModifiedTimeMixerModel, self).__init__()

        if unit_capacity_dict is None:
            unit_capacity_dict = load_unit_capacity_limits()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_seq_len = output_seq_len
        self.input_seq_len = input_seq_len

        self.configs = TimeMixerConfig()
        self.configs.seq_len = input_seq_len
        self.configs.pred_len = output_seq_len
        self.configs.d_model = hidden_dim

        # 共享特征提取部分
        self.feature_adapter = nn.Linear(input_dim, hidden_dim)
        self.pdm_blocks = nn.ModuleList([
            SolarModifiedPDM(self.configs, unit_capacity_dict, skip_weight)
            for _ in range(1)
        ])

        # 功率预测分支
        self.power_predict_layer = nn.Linear(input_seq_len, output_seq_len)
        self.power_projection_layer = nn.Linear(hidden_dim, 1)

        # SSRD预测分支
        self.ssrd_predict_layer = nn.Linear(input_seq_len, output_seq_len)
        self.ssrd_projection_layer = nn.Linear(hidden_dim, 1)

        # 光伏SSRD激活层
        self.solar_activation = SolarSSRDActivation(
            unit_capacity_dict=unit_capacity_dict,
            A=10.0,
            eta=0.2,
            learnable=True,
            balance_constraint=True
        )
        self.skip_weight = nn.Parameter(torch.tensor(skip_weight, dtype=torch.float))

    def _reshape_flat_input(self, x):
        batch_size = x.size(0)
        # 强制 orig_seq_len 与 input_seq_len 匹配
        orig_seq_len = self.input_seq_len  # 使用模型的 input_seq_len (例如 96)

        # 根据 input_seq_len 调整输入特征切片
        expected_features = orig_seq_len * 3 + 3  # power + weather + time + unit_type(2) + unit_id(1)
        if x.size(1) != expected_features:
            raise ValueError(f"Expected input features {expected_features}, got {x.size(1)}")

        power_features = x[:, :orig_seq_len]
        weather_features = x[:, orig_seq_len:2 * orig_seq_len]
        time_features = x[:, 2 * orig_seq_len:3 * orig_seq_len]
        unit_type_features = x[:, 3 * orig_seq_len:3 * orig_seq_len + 2]
        unit_id_features = x[:, -1:]

        features_seq = torch.zeros(batch_size, orig_seq_len, 5, device=x.device)
        features_seq[:, :, 0] = power_features
        features_seq[:, :, 1] = weather_features
        features_seq[:, :, 2] = time_features
        features_seq[:, :, 3] = unit_type_features[:, 0].unsqueeze(1).repeat(1, orig_seq_len)
        features_seq[:, :, 4] = unit_id_features.repeat(1, orig_seq_len)

        is_solar = unit_type_features[:, 1:2]
        unit_ids = unit_id_features

        # 返回重塑后的特征、输入序列中的天气特征（历史天气数据）、是否为太阳能、单元ID
        return features_seq, weather_features, is_solar, unit_ids

    def forward(self, x):
        if x.dim() == 2:
            x, input_weather_data, is_solar, unit_ids = self._reshape_flat_input(x)
        else:
            raise NotImplementedError("暂不支持直接传入3D张量")

        # 共享特征提取
        shared_features = self.feature_adapter(x)

        for pdm in self.pdm_blocks:
            # 这里仍然使用输入的历史天气数据用于特征提取，这是合理的
            shared_features = pdm(shared_features, input_weather_data, is_solar, unit_ids)

        # 功率预测分支
        power_features = shared_features.transpose(1, 2)
        power_features = self.power_predict_layer(power_features)
        power_features = power_features.transpose(1, 2)
        power_raw = self.power_projection_layer(power_features)

        # SSRD预测分支
        ssrd_features = shared_features.transpose(1, 2)
        ssrd_features = self.ssrd_predict_layer(ssrd_features)
        ssrd_features = ssrd_features.transpose(1, 2)
        ssrd_pred = self.ssrd_projection_layer(ssrd_features)

        # 这里是关键改进：使用预测的SSRD来激活功率预测
        # 从预测的SSRD中提取功率预测所需的天气数据
        predicted_weather = ssrd_pred.squeeze(-1)  # [batch_size, output_seq_len]

        # 使用预测的天气数据作为SSRD激活的输入
        power_activated = self.solar_activation(power_raw, predicted_weather, is_solar, unit_ids)

        # 合并原始和激活后的功率
        power_output = self.skip_weight * power_activated + (1.0 - self.skip_weight) * power_raw

        # 返回功率预测和SSRD预测
        return power_output.squeeze(-1), ssrd_pred.squeeze(-1)


# 11. 计算参数量函数
def count_parameters(model):
    """计算模型参数量"""
    # 计算总参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params:,}")

    # 打印每层参数量
    print("\n每层参数详情:")
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            print(f"{name}: {parameter.numel():,}")

    return total_params


def main_solar_only():
    """主函数，训练同时预测功率和SSRD的光伏模型"""
    # 设置参数
    file_path = 'IDunits.csv'  # 数据文件路径
    capacity_file_path = 'MinMax - 副本.xlsx'  # 机组容量文件路径

    batch_size = 64
    num_epochs = 300
    learning_rate = 0.0006
    ssrd_weight = 0.3  # SSRD损失权重

    # 加载数据
    print("加载和处理数据...")
    try:
        # 加载主数据
        df, power_cols, weather_cols = load_data(file_path)

        # 加载机组容量限制数据
        unit_capacity_dict = load_unit_capacity_limits(capacity_file_path)
        if not unit_capacity_dict:
            print("警告：未能加载机组容量数据，将使用默认值")

        # 准备时间序列数据 - 同时准备功率和SSRD目标
        print("准备光伏时间序列数据...")
        data_dict = prepare_time_series_data(
            df, power_cols, weather_cols,
            input_days=1.65625,
            forecast_days=1,
            test_split=0.2,
            slide_by=24
        )

        # 创建数据加载器
        print("创建数据加载器...")
        train_loader, val_loader, test_loader = create_train_val_test_loaders(data_dict, batch_size=batch_size)

        # 计算输出序列长度
        output_seq_len = data_dict['output_seq_len']
        orig_seq_len = (data_dict['train_dataset'].features[0].size(0) - 3) // 3

        # 定义双输出模型
        print("初始化双输出光伏模型(功率+SSRD)...")
        model = SolarModifiedTimeMixerModel(
            input_dim=5,  # 五个特征
            hidden_dim=128,
            output_seq_len=output_seq_len,
            input_seq_len=orig_seq_len,
            unit_capacity_dict=unit_capacity_dict
        )
        model = model.to(device)

        # 计算模型参数量
        count_parameters(model)

        # 验证模型前向传播
        print("验证模型前向传播...")
        try:
            sample_batch = next(iter(train_loader))
            sample_batch_x = sample_batch[0]  # 输入特征
            sample_batch_power = sample_batch[1]  # 功率目标
            sample_batch_ssrd = sample_batch[2]  # SSRD目标

            sample_batch_x = sample_batch_x.to(device)
            sample_batch_power = sample_batch_power.to(device)
            sample_batch_ssrd = sample_batch_ssrd.to(device)

            with torch.no_grad():
                power_output, ssrd_output = model(sample_batch_x)

            print(f"模型功率输出形状: {power_output.shape}, SSRD输出形状: {ssrd_output.shape}")
            print(f"预期功率目标形状: {sample_batch_power.shape}, 预期SSRD目标形状: {sample_batch_ssrd.shape}")
            print("模型前向传播验证成功！")
        except Exception as e:
            print(f"模型前向传播验证失败: {e}")
            import traceback
            traceback.print_exc()
            return

        # 使用MSE损失函数 - 功率和SSRD各自使用一个损失函数
        power_criterion = nn.MSELoss()
        ssrd_criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.8, patience=3, verbose=True
        )

        # 训练模型
        print("开始训练双输出光伏模型...")
        history = train_model(
            model, train_loader, val_loader,
            optimizer, power_criterion, ssrd_criterion, num_epochs,
            device, ssrd_weight, scheduler
        )

        # 加载最佳模型
        print("加载最佳模型...")
        model.load_state_dict(torch.load('best_model.pth'))

        # 评估模型
        print("评估模型...")
        eval_results = evaluate_model_performance(
            model, test_loader, power_criterion, ssrd_criterion,
            device, data_dict, dataset_type='test', ssrd_weight=ssrd_weight
        )

        # 可视化光伏激活层参数
        print("\n===== 光伏激活层学习到的参数 =====")
        for i, pdm in enumerate(model.pdm_blocks):
            print(f"PDM块 {i + 1}:")
            act = pdm.solar_activation
            print(f"  面积(A): {act.A.item():.2f} m²")
            print(f"  效率(eta): {act.eta.item():.4f}")
            print(f"  c_prime: {act.c_prime.item():.4f}")
            print(f"  alpha: {act.alpha.item():.4f}")
            print(f"  alpha_prime: {act.alpha_prime.item():.4f}")

        # 可视化训练历史
        plt.figure(figsize=(15, 10))

        # 总损失
        plt.subplot(2, 2, 1)
        plt.plot(history['train_loss'], label='train_loss')
        plt.plot(history['val_loss'], label='val_loss')
        plt.title('')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # 功率损失
        plt.subplot(2, 2, 2)
        plt.plot(history['train_power_loss'], label='train_power_loss')
        plt.plot(history['val_power_loss'], label='val_power_loss')
        plt.title('')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # SSRD损失
        plt.subplot(2, 2, 3)
        plt.plot(history['train_ssrd_loss'], label='train_ssrd_loss')
        plt.plot(history['val_ssrd_loss'], label='val_ssrd_loss')
        plt.title('')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # 比较功率和SSRD损失
        plt.subplot(2, 2, 4)
        plt.plot(history['train_power_loss'], label='train_power_loss')
        plt.plot(history['train_ssrd_loss'], label='train_ssrd_loss')
        plt.title('')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('dual_output_training_history.png')
        plt.show()

        # 可视化测试集预测结果
        print("\n生成测试集预测可视化...")

        # 选择几个样本进行可视化
        num_samples = min(5, len(eval_results['power_predictions']))
        sample_indices = np.random.choice(len(eval_results['power_predictions']), num_samples, replace=False)

        plt.figure(figsize=(15, 10))
        for i, idx in enumerate(sample_indices):
            unit_id = eval_results['unit_ids'][idx]

            # 功率预测
            plt.subplot(num_samples, 2, i * 2 + 1)
            plt.plot(eval_results['power_targets_original'][idx], label='target_power')
            plt.plot(eval_results['power_predictions_original'][idx], label='prediction_power')
            plt.title(f'Unit {unit_id} power_prediction')
            plt.xlabel('times')
            plt.ylabel('power')
            plt.legend()
            plt.grid(True)

            # SSRD预测
            plt.subplot(num_samples, 2, i * 2 + 2)
            plt.plot(eval_results['ssrd_targets_original'][idx], label='target_SSRD')
            plt.plot(eval_results['ssrd_predictions_original'][idx], label='predction_SSRD')
            plt.title(f'Unit {unit_id} SSRD_prediction')
            plt.xlabel('times')
            plt.ylabel('SSRD')
            plt.legend()
            plt.grid(True)
        import os
        save_dir = './eval_data'
        os.makedirs(save_dir, exist_ok=True)

        # 保存每个样本的数据
        for i, idx in enumerate(sample_indices):
            unit_id = eval_results['unit_ids'][idx]

            # 准备数据
            data = {
                'time_step': range(len(eval_results['power_targets_original'][idx])),
                'power_target': eval_results['power_targets_original'][idx],
                'power_prediction': eval_results['power_predictions_original'][idx],
                'ssrd_target': eval_results['ssrd_targets_original'][idx],
                'ssrd_prediction': eval_results['ssrd_predictions_original'][idx]
            }

            # 创建DataFrame并保存
            df = pd.DataFrame(data)
            filename = f'unit_{unit_id}_sample_{i}.csv'
            filepath = os.path.join(save_dir, filename)
            df.to_csv(filepath, index=False)
            print(f"已保存: {filepath}")

        # 保存元数据
        metadata = {
            'sample_count': len(sample_indices),
            'unit_ids': [eval_results['unit_ids'][idx] for idx in sample_indices],
            'sample_indices': sample_indices.tolist()
        }
        metadata_df = pd.DataFrame([metadata])
        metadata_path = os.path.join(save_dir, 'metadata.csv')
        metadata_df.to_csv(metadata_path, index=False)
        print(f"已保存元数据: {metadata_path}")
        print(f"所有数据已保存到目录: {save_dir}")
        plt.tight_layout()
        plt.savefig('dual_output_test_predictions.svg',format='svg')
        plt.show()

        return model, eval_results, history, data_dict

    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_and_plot(plot_type='line'):
    """加载保存的数据并绘图"""
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    save_dir = './eval_data'

    # 读取元数据
    metadata = pd.read_csv(os.path.join(save_dir, 'metadata.csv')).iloc[0]
    unit_ids_str = metadata['unit_ids'].strip('[]').replace("'", "").split(', ')
    unit_ids = [int(uid) for uid in unit_ids_str]

    # 加载数据
    loaded_data = []
    for i, unit_id in enumerate(unit_ids):
        filename = f'unit_{unit_id}_sample_{i}.csv'
        filepath = os.path.join(save_dir, filename)
        df = pd.read_csv(filepath)
        loaded_data.append({
            'unit_id': unit_id,
            'power_target': df['power_target'].values,
            'power_prediction': df['power_prediction'].values,
            'ssrd_target': df['ssrd_target'].values,
            'ssrd_prediction': df['ssrd_prediction'].values
        })

    # 绘图
    num_samples = len(loaded_data)

    if plot_type == 'line':
        plt.figure(figsize=(15, 10))
        for i, data in enumerate(loaded_data):
            # 功率预测
            plt.subplot(num_samples, 2, i * 2 + 1)
            plt.plot(data['power_target'], label='target_power')
            plt.plot(data['power_prediction'], label='prediction_power')
            plt.title(f'Unit {data["unit_id"]} power_prediction')
            plt.xlabel('times')
            plt.ylabel('power')
            plt.legend()
            plt.grid(True)

            # SSRD预测
            plt.subplot(num_samples, 2, i * 2 + 2)
            plt.plot(data['ssrd_target'], label='target_SSRD')
            plt.plot(data['ssrd_prediction'], label='prediction_SSRD')
            plt.title(f'Unit {data["unit_id"]} SSRD_prediction')
            plt.xlabel('times')
            plt.ylabel('SSRD')
            plt.legend()
            plt.grid(True)

    elif plot_type == 'radar':
        plt.figure(figsize=(15, 10))
        for i, data in enumerate(loaded_data):
            n_times = len(data['power_target'])
            angles = np.linspace(0, 2 * np.pi, n_times, endpoint=False).tolist()
            angles += angles[:1]  # 闭合

            # 功率预测雷达图
            ax1 = plt.subplot(num_samples, 2, i * 2 + 1, projection='polar')
            power_target = data['power_target'].tolist() + [data['power_target'][0]]
            power_pred = data['power_prediction'].tolist() + [data['power_prediction'][0]]

            ax1.plot(angles, power_target, 'o-', linewidth=2, label='target_power', color='blue')
            ax1.plot(angles, power_pred, 'o-', linewidth=2, label='prediction_power', color='orange')
            ax1.fill(angles, power_target, alpha=0.2, color='blue')
            ax1.fill(angles, power_pred, alpha=0.2, color='orange')

            ax1.set_xticks(angles[:-1])
            ax1.set_xticklabels([f'{i}' for i in range(n_times)])
            ax1.set_title(f'Unit {data["unit_id"]} power_prediction')
            ax1.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
            ax1.grid(True)

            # SSRD预测雷达图
            ax2 = plt.subplot(num_samples, 2, i * 2 + 2, projection='polar')
            ssrd_target = data['ssrd_target'].tolist() + [data['ssrd_target'][0]]
            ssrd_pred = data['ssrd_prediction'].tolist() + [data['ssrd_prediction'][0]]

            ax2.plot(angles, ssrd_target, 'o-', linewidth=2, label='target_SSRD', color='blue')
            ax2.plot(angles, ssrd_pred, 'o-', linewidth=2, label='prediction_SSRD', color='orange')
            ax2.fill(angles, ssrd_target, alpha=0.2, color='blue')
            ax2.fill(angles, ssrd_pred, alpha=0.2, color='orange')

            ax2.set_xticks(angles[:-1])
            ax2.set_xticklabels([f'{i}' for i in range(n_times)])
            ax2.set_title(f'Unit {data["unit_id"]} SSRD_prediction')
            ax2.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
            ax2.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 执行主函数
    main_solar_only()
    # load_and_plot('radar')