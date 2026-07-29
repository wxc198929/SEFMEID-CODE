import os
import numpy as np
import scipy.io as sio
import h5py

# =============================
# 参数配置
# =============================
# 批量处理 24021 到 24055
subject_ids = [f"240{i:02d}" for i in range(21, 56)]  # ['24021', '24022', ..., '24055']

base_snirf_dir = r"...\SEFMID\Raw data\fNIRS_snirf" # 选择原始转换文件路径
base_mat_dir = r"...SEFMID\Processed data\fNIRS" # 选择预处理后文件路径
save_dir = r"...SEFMID\Segment data\fNIRS" # 选择保存划分文件路径
os.makedirs(save_dir, exist_ok=True)

sfreq = 11                 # fNIRS 采样率 Hz
segment_length_s = 8       # 每段长度（秒）
target_length = int(segment_length_s * sfreq)  # 必须为 88
max_segments = 200         # 最多保留段数

# =============================
# 批量循环处理
# =============================
for sid in subject_ids:
    print(f"\n============================")
    print(f"正在处理编号: {sid}")
    print(f"============================")

    snirf_path = os.path.join(base_snirf_dir, f"{sid}.snirf")
    mat_file = os.path.join(base_mat_dir, f"{sid}.mat")
    save_file = os.path.join(save_dir, f"{sid}_segments_fnirs.mat")

    if not os.path.exists(snirf_path):
        print(f"找不到 SNIRF 文件: {snirf_path}，跳过该受试者")
        continue
    if not os.path.exists(mat_file):
        print(f"找不到 MAT 文件: {mat_file}，跳过该受试者")
        continue

    # =============================
    # 读取 SNIRF 事件
    # =============================
    with h5py.File(snirf_path, 'r') as f:
        stim1 = f['nirs']['stim1']
        stim_data = stim1['data'][:]
        onset_times = np.round(stim_data, 5)

    print(f"事件时间点（秒）：{onset_times}")

    # =============================
    # 读取预处理 fNIRS 数据
    # =============================
    mat_data = sio.loadmat(mat_file)
    data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
    if 'data' in mat_data:
        data = mat_data['data']
    else:
        data = mat_data[data_keys[0]]

    n_channels, n_times = data.shape
    print(f"fNIRS 数据形状: {data.shape} (通道 x 时间点)")

    # =============================
    # 按事件划分并强制每段为 88 个点
    # =============================
    segment_list = []

    for i, onset in enumerate(onset_times):
        if len(segment_list) >= max_segments:
            print("已达到最大段数限制，停止划分。")
            break

        start_sample = int(np.round(onset * sfreq))
        end_sample = start_sample + target_length

        # 提取片段（如果超出末尾，取到末尾）
        if end_sample <= n_times:
            seg = data[:, start_sample:end_sample]
        else:
            seg = data[:, start_sample:]

        current_len = seg.shape[1]

        if current_len != target_length:
            print(f"警告: 段{i+1} 原始长度 {current_len} ≠ 88，正在调整...")

            if current_len < target_length:
                # 用最后一个时间点重复填充
                pad_width = target_length - current_len
                last_point = seg[:, -1:]  # 最后一列 (40, 1)
                padding = np.tile(last_point, (1, pad_width))  # (40, pad_width)
                seg = np.concatenate([seg, padding], axis=1)
                print(f"  → 填充至 88 点")

            elif current_len > target_length:
                # 截断前 88 点
                seg = seg[:, :target_length]
                print(f"  → 截断至 88 点")

        segment_list.append(seg)
        print(f"段{i+1}: 起点 {start_sample}, 最终长度 {seg.shape[1]}")

    print(f"该受试者共保留 {len(segment_list)} 段，每段形状 (40, 88)")

    # =============================
    # 保存为 MATLAB cell 数组
    # =============================
    segments_cell = np.empty(len(segment_list), dtype=object)
    for i, seg in enumerate(segment_list):
        segments_cell[i] = seg

    sio.savemat(save_file, {
        'segments': segments_cell,
        'sfreq': sfreq,
        'segment_length_s': segment_length_s,
        'target_points': target_length
    })
    print(f"已保存: {save_file}")

print("\n所有受试者（24021-24055）批量处理完成！")