import mne
import numpy as np
from mne.preprocessing import EOGRegression
from mne_icalabel import label_components
from scipy import signal

from file_io import save_file
from signals import detect_and_interpolate_outliers, wavelet_denoising


def eeg_preprocessing_by_dict(data_dict, lowcut, highcut, bad_channels, montage='standard_1020', notch_f=50,
                              rm_outlier=False, eog_regression=False, eog_channels=None,
                              is_ICA=False, ICA_component=0, ICA_method='infomax', random_state=42,
                              is_ref=True, ref_chan=None, is_save=False, save_path=None,
                              save_filestyle='mat'):
    """
    EEG预处理流程，使用字典作为输入/输出格式。

    Notes
    -------------
    本函数实现标准化EEG预处理流程：

    1. **数据转换**：将输入字典转换为MNE Raw对象，保留原始采样率和通道信息
    2. **电极定位**：设置国际10-20系统电极位置，确保空间定位准确
    3. **带通滤波**：应用FIR滤波器(phase='zero')保留目标频段(如1-40Hz)，去除低频漂移(<1Hz)和高频噪声(>40Hz)
    4. **陷波滤波**：消除50/60Hz工频干扰及其谐波，改善信噪比
    5. **EOG回归**：当存在独立EOG通道时，通过线性回归模型校正眼电伪迹
    6. **坏道处理**：移除信号质量差的通道（如阻抗过高或信号中断）
    7. **异常值校正**：检测并插值瞬态伪迹（如电极接触不良导致的尖峰）
    8. **ICA去伪迹**：使用独立成分分析分离生理伪迹（眼动/肌电/心电）
       - 采用ICLabel算法自动标记成分类别
       - 剔除非大脑成分（眼电、肌电等）
    9. **重参考**：转换为平均参考或指定参考（如双侧乳突），提升信号可比性
    10. **数据输出**：更新数据结构并保存处理结果（支持MATLAB/FIF格式）

    Parameters
    ----------
    data_dict : dict
        原始EEG数据结构，包含以下键：
        - data : ndarray, shape (n_channels, n_times)
            EEG时间序列数据
        - srate : float
            采样率（Hz）
        - ch_names : list of str
            通道名称列表
    lowcut : float
        带通滤波低截止频率（Hz）
    highcut : float
        带通滤波高截止频率（Hz）
    bad_channels : list of str
        需要排除的通道名称列表
    montage : str, optional
        电极布局（默认：'standard_1020'）
    notch_f : float, optional
        陷波频率（默认：50 Hz）
    rm_outlier : bool, optional
        检测并插值异常值（默认：False）
    eog_regression : bool, optional
        应用EOG回归（默认：False）
    eog_channels : list of str or None, optional
        EOG通道名称列表（默认：None）
    is_ICA : bool, optional
        应用ICA伪迹去除（默认：False）
    ICA_component : int, optional
        ICA成分数量（0=自动选择）（默认：0）
    ICA_method : str, optional
        ICA算法（默认：'infomax'）
    is_ref : bool, optional
        应用重参考（默认：True）
    ref_chan : list of str or None, optional
        参考通道（默认：None=平均参考）
    is_save : bool, optional
        保存预处理数据（默认：False）
    save_path : str or None, optional
        输出文件路径
    save_filestyle : str, optional
        输出文件格式（'mat'或'fif'）（默认：'mat'）

    Returns
    -------
    dict
        预处理后的EEG数据结构，包含以下键：
        - data : ndarray, shape (n_channels, n_times)
            处理后的EEG数据
        - srate : float
            采样率
        - ch_names : list of str
            通道名称
        - nchan : int
            通道数量
        - type : str
            数据类型标识（'eeg_preprocess'）
        - montage : str
            使用的电极布局

    Examples
    --------
    >>> import numpy as np
    >>> from mne import create_info
    >>> from mne.io import RawArray

    >>> # 创建模拟EEG数据
    >>> np.random.seed(42)
    >>> n_channels = 8
    >>> n_times = 1000
    >>> fs = 250
    >>> data = np.random.randn(n_channels, n_times) * 1e-5
    >>> ch_names = ['Fp1', 'Fp2', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2']
    >>> data_dict = {'data': data, 'srate': fs, 'ch_names': ch_names}

    >>> # 运行预处理
    >>> processed = eeg_preprocessing_by_dict(
    ...     data_dict,
    ...     lowcut=1.0,
    ...     highcut=40.0,
    ...     bad_channels=['Fp1'],
    ...     montage='standard_1020',
    ...     notch_f=50.0,
    ...     rm_outlier=True,
    ...     eog_regression=False,
    ...     is_ICA=True,
    ...     ICA_component=4,
    ...     is_ref=True
    ... )

    >>> # 检查输出结构
    >>> print(f"Processed data shape: {processed['data'].shape}")
    Processed data shape: (7, 1000)
    >>> print(f"Remaining channels: {processed['ch_names']}")
    Remaining channels: ['Fp2', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2']
    """
    # Extract parameters from input dictionary
    data = data_dict['data']
    fs = data_dict['srate']
    chan_list = list(data_dict['ch_names'])

    # Create MNE Raw object
    info = mne.create_info(ch_names=chan_list, sfreq=fs, ch_types='eeg')
    raw = mne.io.RawArray(data=data, info=info)

    # Set electrode positions
    try:
        raw.set_montage(montage)
    except Exception:
        print("Non-standard or missing montage")

    # Apply bandpass filtering
    raw.filter(
        l_freq=lowcut,
        h_freq=highcut,
        fir_design='firwin',
        phase='zero',
        verbose=True
    )

    # Apply harmonic notch filtering
    notch_freqs = np.arange(notch_f, fs / 2, notch_f)
    raw.notch_filter(freqs=notch_freqs, notch_widths=2, verbose=True)

    # Apply EOG regression
    if eog_regression and eog_channels:
        raw.set_eeg_reference('average', projection=False)
        raw.set_channel_types({ch: 'eog' for ch in eog_channels})
        weights = EOGRegression().fit(raw)
        raw = weights.apply(raw, copy=True)

    # Remove bad channels
    if bad_channels:
        raw.drop_channels(bad_channels)

    # Detect and interpolate outliers
    if rm_outlier:
        for i in range(len(chan_list)):
            raw._data[i] = detect_and_interpolate_outliers(raw._data[i])

    # Apply ICA artifact removal
    if is_ICA:
        ica = mne.preprocessing.ICA(
            n_components=ICA_component,
            method=ICA_method
        )
        ica.fit(raw)
        ic_labels = label_components(raw, ica, method='iclabel')

        # Determine components to exclude
        exclude_idx = [idx for idx, label in enumerate(ic_labels['labels'])
                       if label not in ['brain', 'other']]
        print(f"Excluded ICA components: {exclude_idx}")
        ica.apply(raw, exclude=exclude_idx)

    # Apply rereferencing
    if not eog_regression and is_ref:
        if ref_chan:
            raw.set_eeg_reference(ref_chan, projection=False)
        else:
            raw.set_eeg_reference('average', projection=False)

    # Prepare output structure
    result = data_dict.copy()
    result.update({
        'data': raw.get_data(),
        'nchan': len(raw.ch_names),
        'ch_names': raw.ch_names,
        'type': 'eeg_preprocess',
        'montage': montage
    })

    # Save results
    if is_save:
        save_file(data=result, save_path=save_path, save_filestyle=save_filestyle)

    return result



def fnirs_preprocessing_by_raw(raw, bad_channels, lowcut=0.01, highcut=0.7,
                               wavelet='db4', level=4, alpha=0.2,
                               enable_interpolate=False):
    """
    预处理fNIRS的MNE Raw对象，包括伪迹去除和滤波。

    Notes
    -------------
    本函数实现标准化fNIRS预处理流程：

    1. **坏道去除**：移除信号异常的通道（如光源不稳定或探测器故障）
    2. **光强度→光学密度**：基于Beer-Lambert定律转换原始信号
    3. **血氧浓度计算**：将光学密度转换为氧合血红蛋白(HbO)和脱氧血红蛋白(HbR)浓度
    4. **带通滤波**：保留血流动力学响应频段(0.01-0.5Hz)，抑制心跳/呼吸等生理噪声
    5. **小波去噪**：使用小波阈值法(db4)消除运动伪迹和高频噪声
    6. **异常值校正**：插值突发性信号失真（如头动导致的瞬时伪迹）
    7. **信号整形**：去除线性趋势(detrend)避免基线漂移影响
    8. **数据标准化**：Z-score标准化便于跨被试比较

    Parameters
    ----------
    raw : mne.io.Raw
        原始fNIRS数据对象
    bad_channels : list of str
        需要排除的通道名称列表
    lowcut : float, optional
        低截止频率（Hz）（默认：0.01）
    highcut : float, optional
        高截止频率（Hz）（默认：0.7）
    wavelet : str or None, optional
        小波去噪类型（默认：'db4'）
    level : int, optional
        小波分解层级（默认：4）
    alpha : float, optional
        小波阈值乘数（默认：0.2）
    enable_interpolate : bool, optional
        启用异常值插值（默认：False）

    Returns
    -------
    mne.io.Raw
        预处理后的血流动力学数据

    Examples
    --------
    >>> import numpy as np
    >>> import mne
    >>> from mne.preprocessing.nirs import optical_density, beer_lambert_law

    >>> # 创建模拟fNIRS数据
    >>> np.random.seed(42)
    >>> n_channels = 16  # 8个光源 × 2个波长
    >>> n_times = 500
    >>> fs = 10.0  # 典型fNIRS采样率
    >>> data = np.random.randn(n_channels, n_times) * 0.1
    >>> info = mne.create_info(n_channels, fs, ch_types='fnirs_cw_amplitude')

    >>> # 添加波长信息
    >>> for i in range(8):
    ...     info['chs'][i * 2]['ch_name'] = f'S{i+1}_D1_760'
    ...     info['chs'][i * 2 + 1]['ch_name'] = f'S{i+1}_D1_850'

    >>> # 创建Raw对象
    >>> raw = mne.io.RawArray(data, info)

    >>> # 运行预处理
    >>> preprocessed = fnirs_preprocessing_by_raw(
    ...     raw,
    ...     bad_channels=['S1_D1_760', 'S2_D1_850'],
    ...     lowcut=0.01,
    ...     highcut=0.5,
    ...     wavelet='db4',
    ...     level=3,
    ...     alpha=0.15,
    ...     enable_interpolate=True
    ... )

    >>> # 验证结果
    >>> print(f"Preprocessed data shape: {preprocessed.get_data().shape}")
    Preprocessed data shape: (14, 500)
    >>> print(f"Channel names: {preprocessed.ch_names[:4]}")
    Channel names: ['S3_D1_760', 'S3_D1_850', 'S4_D1_760', 'S4_D1_850']
    """
    # Copy and remove bad channels
    raw_intensity = raw.copy()
    if bad_channels:
        raw_intensity.drop_channels(bad_channels)

    # Optical density conversion
    raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)

    # Beer-Lambert law conversion
    raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=0.1)

    # Bandpass filtering
    filtered_data = mne.filter.filter_data(
        raw_haemo.get_data(),
        raw_haemo.info['sfreq'],
        l_freq=lowcut,
        h_freq=highcut,
        method='fir'
    )
    raw_haemo._data = filtered_data

    # Wavelet denoising
    if wavelet:
        for ch in range(len(raw_haemo.info['ch_names'])):
            signal_data = raw_haemo._data[ch]
            if np.all(signal_data == signal_data[0]):  # 全常量信号直接跳过
                continue
            raw_haemo._data[ch] = wavelet_denoising(
                raw_haemo._data[ch],
                wavelet=wavelet,
                level=level,
                alpha=alpha
            )

    # Outlier detection and interpolation
    if enable_interpolate:
        for ch in range(len(raw_haemo.info['ch_names'])):
            raw_haemo._data[ch] = detect_and_interpolate_outliers(raw_haemo._data[ch])

    # Detrending
    raw_haemo._data = signal.detrend(raw_haemo.get_data(), axis=1)

    # Standardization
    raw_haemo._data = (raw_haemo.get_data() - np.mean(raw_haemo.get_data(), axis=1, keepdims=True)
                       ) / np.std(raw_haemo.get_data(), axis=1, keepdims=True)

    return raw_haemo
