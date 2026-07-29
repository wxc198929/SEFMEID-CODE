import os
import numpy as np
import scipy.io as sio
import pyedflib

# -----------------------------
# 手动输入要处理的编号
# -----------------------------
subject_ids = [24001]  #  在这里输入要处理的编号（可修改）

# -----------------------------
# 通用参数
# -----------------------------
base_mat_dir = r"...\SEFMID\Processed data\EEG" # 选择预处理后文件路径
base_bdf_dir = r"...\SEFMID\Raw data\EEG" # 选择事件打标文件路径
save_dir = r"...\SEFMID\Segment data\EEG" # 选择保存划分文件路径
os.makedirs(save_dir, exist_ok=True)

sfreq = 1000      # 采样率 Hz
segment_length_s = 8  # 每段长度（秒）
segment_length = int(segment_length_s * sfreq)  # 转换为采样点数
max_segments = 200  #   只保留前200段

# -----------------------------
# 逐个编号处理
# -----------------------------
for sid in subject_ids:
    print(f"\n============================")
    print(f"正在处理编号: {sid}")
    print(f"============================")

    mat_file = os.path.join(base_mat_dir, f"{sid}.mat")
    bdf_file = os.path.join(base_bdf_dir, str(sid), "evt.bdf")

    if not os.path.exists(mat_file):
        print(f"找不到 MAT 文件: {mat_file}")
        continue
    if not os.path.exists(bdf_file):
        print(f"找不到 BDF 文件: {bdf_file}")
        continue

    # -----------------------------
    # 读取 EEG 数据
    # -----------------------------
    mat_data = sio.loadmat(mat_file)
    data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
    if 'data' in mat_data:
        data = mat_data['data']
    else:
        data = mat_data[data_keys[0]]

    n_channels, n_times = data.shape
    print(f"数据形状: {data.shape}, 通道数: {n_channels}, 时间点数: {n_times}")

    # -----------------------------
    # 读取 BDF 注释
    # -----------------------------
    with pyedflib.EdfReader(bdf_file) as f:
        annotations = f.readAnnotations()

    onsets, durations, descriptions = annotations
    onsets = np.array(onsets)
    descriptions = np.array(descriptions)
    n_segments = len(onsets)
    print(f"总共检测到 {n_segments} 个标记点, 每段取 {segment_length_s} 秒 ({segment_length} 采样点)")

    # -----------------------------
    # 划分数据：以每个标记点为起点，取8秒片段
    # -----------------------------
    segment_list = []
    segment_descriptions = []

    for i in range(n_segments):
        if len(segment_list) >= max_segments:
            print(f"已达到最大段数 {max_segments}，停止提取。")
            break

        start_sample = int(np.round(onsets[i] * sfreq))
        end_sample = start_sample + segment_length

        # 防止越界
        if end_sample > n_times:
            print(f"段 {i+1} 超出数据长度，跳过。")
            continue

        seg = data[:, start_sample:end_sample]
        desc = descriptions[i]
        segment_list.append(seg)
        segment_descriptions.append(desc)

        print(f"段 {i+1}: 起点 {start_sample}, 终点 {end_sample}, 描述: {desc}")

    print(f"共保留 {len(segment_list)} 段（最多200段），每段 {segment_length_s} 秒。")

    # -----------------------------
    # 保存结果（只保留前200段）
    # -----------------------------
    segments_cell = np.empty(len(segment_list), dtype=object)
    for i, seg in enumerate(segment_list):
        segments_cell[i] = seg

    save_file = os.path.join(save_dir, f"{sid}_segments_eeg.mat")
    sio.savemat(save_file, {
        'segments': segments_cell,
        'descriptions': segment_descriptions,
        'sfreq': sfreq,
        'segment_length_s': segment_length_s
    })
    print(f"保存划分数据: {save_file}")

print("\n所有指定编号处理完成！")
