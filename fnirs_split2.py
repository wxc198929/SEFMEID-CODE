import os
import numpy as np
import pandas as pd
import scipy.io as sio

# -----------------------------
# 参数设置
# -----------------------------
excel_path = r"...\SEFMID\Raw data\fNIRS_1_19.xlsx" # 选择调整后事件打标文件
base_mat_dir = r"...\SEFMID\Processed data\fNIRS" # 选择预处理后文件路径
save_dir = r"...\SEFMID\Segment data\fNIRS" # 选择保存划分文件路径
os.makedirs(save_dir, exist_ok=True)

sfreq = 11             # fNIRS 采样率 Hz
segment_length_s = 8   # 每段时长（秒）
segment_length = int(segment_length_s * sfreq)
max_segments = 200     # 最多保存200段

# -----------------------------
# 读取 Excel 文件
# -----------------------------
ext = os.path.splitext(excel_path)[1].lower()
if ext in ['.xls', '.xlsx']:
    df = pd.read_excel(excel_path)
elif ext == '.csv':
    df = pd.read_csv(excel_path, encoding='gbk')
else:
    raise ValueError(f"不支持的文件类型: {ext}")

required_cols = {'subject', 'onset_time_s'}
if not required_cols.issubset(df.columns):
    raise ValueError(f"Excel 文件缺少必要列，必须包含: {required_cols}")

print(f"已读取打标文件: {excel_path}")
print(f"共 {len(df)} 条标记记录。")
print(df.head())

# -----------------------------
# 按 subject 分组处理
# -----------------------------
for subject, group in df.groupby('subject'):
    print(f"\n============================")
    print(f"正在处理受试者: {subject}")
    print(f"============================")

    mat_file = os.path.join(base_mat_dir, f"{int(subject)}.mat")
    if not os.path.exists(mat_file):
        print(f"找不到 fNIRS 文件: {mat_file}")
        continue

    # 读取 fNIRS 数据
    mat_data = sio.loadmat(mat_file)
    data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
    if 'data' in mat_data:
        data = mat_data['data']
    else:
        data = mat_data[data_keys[0]]

    n_channels, n_times = data.shape
    print(f"fNIRS 数据形状: {data.shape}, 通道数: {n_channels}, 时间点数: {n_times}")

    # -----------------------------
    # 划分数据：每个 onset 取 8 秒
    # -----------------------------
    segment_list = []

    for i, row in group.iterrows():
        if len(segment_list) >= max_segments:
            print(f"已达到最大段数 {max_segments}，停止提取。")
            break

        onset = float(row['onset_time_s'])
        start_sample = int(np.round(onset * sfreq))
        end_sample = start_sample + segment_length

        if end_sample > n_times:
            print(f"跳过段 {i}：超出数据长度 ({end_sample} > {n_times})")
            continue

        seg = data[:, start_sample:end_sample]
        segment_list.append(seg)
        print(f"段 {len(segment_list)}: onset={onset}s, shape={seg.shape}")

    print(f"共保留 {len(segment_list)} 段（每段 {segment_length_s}s）")

    # -----------------------------
    # 保存结果
    # -----------------------------
    if len(segment_list) > 0:
        segments_cell = np.empty(len(segment_list), dtype=object)
        for i, seg in enumerate(segment_list):
            segments_cell[i] = seg

        save_file = os.path.join(save_dir, f"{int(subject)}_segments_fnirs.mat")
        sio.savemat(save_file, {
            'segments': segments_cell,
            'sfreq': sfreq,
            'segment_length_s': segment_length_s
        })
        print(f"保存文件: {save_file}")
    else:
        print(f"未提取到有效段，跳过保存。")

print("\n所有受试者处理完成！")
