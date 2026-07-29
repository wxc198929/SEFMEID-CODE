import os
from file_io import read_file
from preprocess import eeg_preprocessing_by_dict
import numpy as np

def batch_process_eeg(root_path, save_root, mode="auto", subject_ids=None):
    """
    批量处理 EEG 数据
    mode: "auto" 自动遍历全部文件夹；"manual" 手动选择文件夹编号
    subject_ids: 当 mode="manual" 时，指定如 ["24001", "24005"]
    """
    np.random.seed(42)

    # ===========================
    # 选择文件夹模式
    # ===========================
    if mode == "auto":
        # 自动遍历根目录下所有文件夹
        subject_folders = [f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))]
        print(f"自动模式：共找到 {len(subject_folders)} 个文件夹\n")
    elif mode == "manual":
        if subject_ids is None or len(subject_ids) == 0:
            raise ValueError("手动模式下必须提供 subject_ids，例如 ['24001', '24002']")
        subject_folders = subject_ids
        print(f"手动模式：将处理 {len(subject_folders)} 个指定文件夹\n")
    else:
        raise ValueError("mode 参数只能是 'auto' 或 'manual'")

    # ===========================
    # 遍历文件夹并处理
    # ===========================
    for folder in subject_folders:
        subject_path = os.path.join(root_path, folder)
        bdf_file = os.path.join(subject_path, 'data.bdf')

        if not os.path.exists(bdf_file):
            print(f"未找到文件：{bdf_file}，跳过")
            continue

        print(f"正在处理：{bdf_file}")

        try:
            # 读取原始数据
            data = read_file([bdf_file])

            # 保存路径
            save_path = os.path.join(save_root, f"{folder}.mat")

            # EEG 预处理
            eeg_preprocessing_by_dict(
                data_dict=data,
                lowcut=1.0,
                highcut=40.0,
                bad_channels=[],
                montage='standard_1020',
                notch_f=50,
                rm_outlier=True,
                eog_regression=False,
                eog_channels=None,
                is_ICA=True,
                ICA_component=0.99,
                ICA_method='infomax',
                is_ref=True,
                ref_chan=[],
                is_save=True,
                save_path=save_path
            )

            print(f"{folder} 处理完成，已保存到 {save_path}\n")

        except Exception as e:
            print(f"处理 {folder} 时出错：{e}\n")


if __name__ == '__main__':
    root_eeg_path = r"...\SEFMID\Raw data\EEG" # 选择原始数据路径
    save_dir = r"...SEFMID\Processed data\EEG" # 选择保存预处理文件路径
    os.makedirs(save_dir, exist_ok=True)

    # ===========================
    # 设置模式选项
    # ===========================
    # mode 可选：
    #   "auto"  -> 自动遍历所有文件夹
    #   "manual" -> 手动指定要处理的编号
    mode = "auto"  # ← 改这里切换模式

    # 当 mode="manual" 时手动指定编号
    subject_ids = ["24001"]

    # 运行批处理
    batch_process_eeg(root_eeg_path, save_dir, mode=mode, subject_ids=subject_ids)
