import os
import mne
import scipy.io as sio
from preprocess import fnirs_preprocessing_by_raw 
import tkinter as tk
from tkinter import filedialog

# ---------------------- 配置路径 ----------------------
snirf_dir = r"...\SEFMID\Raw data\fNIRS_snirf" # 选择格式转换后的fnirs文件夹
save_dir = r"...SEFMID\Processed data\fNIRS" # 选择预处理后保存路径

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# ---------------------- 选择模式 ----------------------
# mode = 1 : 遍历整个文件夹
# mode = 2 : 手动选择单个文件
mode = 2

# ---------------------- 获取待处理文件列表 ----------------------
file_list = []

if mode == 1:
    # 遍历整个文件夹中所有 .snirf 文件
    file_list = [
        os.path.join(snirf_dir, f)
        for f in os.listdir(snirf_dir)
        if f.endswith(".snirf")
    ]

elif mode == 2:
    # 手动选择单个或多个文件
    root = tk.Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(
        title="选择一个或多个 .snirf 文件",
        filetypes=[("SNIRF files", "*.snirf")]
    )
    if file_paths:
        file_list = list(file_paths)
    else:
        print("未选择文件，程序已退出。")
        exit()

else:
    print("输入错误，请输入 1 或 2。")
    exit()

# ---------------------- 开始处理 ----------------------
for file_path in file_list:
    file_name = os.path.basename(file_path)
    print(f"正在读取: {file_path}")

    try:
        # 读取原始 SNIRF 文件
        raw = mne.io.read_raw_snirf(file_path, preload=True)

        # 调用 preprocess.py 中的函数进行预处理
        raw_preprocessed = fnirs_preprocessing_by_raw(
            raw=raw,
            bad_channels=[],       # 可根据需要指定坏道
            lowcut=0.01,
            highcut=0.2,
            wavelet='db4',
            level=4,
            alpha=0.3,
            enable_interpolate=True
        )

        # 提取数据
        data, times = raw_preprocessed.get_data(return_times=True)
        info = raw_preprocessed.info

        # 保存为 .mat 文件
        mat_data = {
            "data": data,
            "times": times,
            "ch_names": info['ch_names'],
            "sfreq": info['sfreq']
        }

        filename_no_ext = os.path.splitext(file_name)[0]
        save_path = os.path.join(save_dir, filename_no_ext + ".mat")
        sio.savemat(save_path, mat_data)
        print(f"已保存: {save_path}")

    except Exception as e:
        print(f"处理失败 {file_name}: {e}")

print("所有文件处理完成！")
