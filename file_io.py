import json
import os
from tkinter import Tk, filedialog
import numpy as np
from scipy.io import loadmat, savemat
import mne
import pyedflib
import snirf



def read_file(path=None):
    """
        读取多种神经影像数据格式文件。

        Parameters
        ----------
        path : str, optional
            文件路径。如果为None，会自动打开文件选择对话框
            支持格式：.edf, .bdf, .snirf, .json, .mat

        Returns
        -------
        dict or None
            包含以下键的数据字典：

            - **data**: 时间序列数据（列表/数组）
            - **srate**: 采样率（浮点数）
            - **events**: 事件标记（[时间, 时长, 标签]组成的列表）
            - **nchan**: 通道数量（整数）
            - **ch_names**: 通道名称（字符串列表）
            - **type**: 数据类型（'eeg'或'fnirs'）
            如果未选择有效文件则返回None

        Examples:
        ------
        >>> data = read_file()  # 打开文件选择对话框
        >>> print(data.keys())  # 输出数据字典的键
        dict_keys(['data', 'srate', 'events', 'nchan', 'ch_names', 'type'])
        >>> # 读取SNIRF文件
        >>> fnirs_data = read_file("subj01_task01.snirf")
        >>> print(fnirs_data['wavelengths'])  # 输出波长信息
        [690.0, 830.0]
        """
    if path is None:
        root = Tk()
        root.withdraw()
        try:
            # select bdf or edf file
            path = filedialog.askopenfilenames(initialdir='/', title='Select one file',
                                               filetypes=(
                                                   ("all files", "*"),
                                                   ("one json file", "*.json"), ("one mat file", "*.mat"),
                                                   ("one edf file", "*.edf"), ("one bdf file", "*.bdf"),
                                                   ("one nirs file", "*.nirs")))
        except Exception as e:
            print(e)
    data = None
    if len(path) == 1:
        file_type = path[0].split('.')[-1]
        if file_type == 'edf':
            data = read_edf(path[0])
        elif file_type == 'bdf':
            data = read_bdf(path[0])
        elif file_type == 'snirf':
            data = read_snirf(path[0])
        elif file_type == 'json':
            data = read_json(path[0])
        elif file_type == 'mat':
            data = read_mat(path[0])

    return data


def read_mat(path=None):
    """
        读取MATLAB(.mat)格式文件并进行数据结构优化。

        Parameters
        ----------
        path : str, optional
            .mat文件路径。如果为None，会打开文件选择对话框

        Returns
        -------
        dict
            经过优化处理的MATLAB数据结构：

            - **data**: 时间序列数据（列表/数组）
            - **srate**: 采样率（浮点数）
            - **events**: 事件标记（[时间, 时长, 标签]组成的列表）
            - **nchan**: 通道数量（整数）
            - **ch_names**: 通道名称（字符串列表）
            - **type**: 数据类型（'eeg'或'fnirs'）

        Notes
        -----
        自动执行以下优化：
        1. 压缩单元素维度
        2. 0维数组转为标量
        3. 嵌套数组转为列表
        4. EEG元数据特殊处理

        Examples:
        ------
        >>> eeg_data = read_mat("eeg_recording.mat")
        >>> print(eeg_data['ch_names'])  # 输出通道名称
        ['Fp1', 'Fp2', 'Cz', 'Pz']
        >>> # 查看事件标记
        >>> for event in eeg_data['events']:
        >>>     print(f"时间: {event[0]}s, 类型: {event[2]}")
        时间: 12.5s, 类型: stimulus
        时间: 25.7s, 类型: response
        """
    def remove_keys(d, char):
        return {k: v for k, v in d.items() if char not in k}

    def reduce_dimensions(arr):
        num_dims = arr.ndim
        num_elements = np.prod(arr.shape)
        if num_elements <= num_dims:
            arr = np.squeeze(arr)
        if arr.ndim == 0:
            arr = arr.item()
        return arr

    def convert_to_list(data):
        if isinstance(data, list) or isinstance(data, np.ndarray):
            if isinstance(data, np.ndarray):
                data = data.tolist()
            return [convert_to_list(item) for item in data]
        else:
            return data

    if path is None:
        root = Tk()
        root.withdraw()
        try:
            # select bdf or edf file
            path = filedialog.askopenfilenames(initialdir='/', title='Select one mat files',
                                               filetypes=(("one mat file", "*.mat"),))[0]
        except Exception as e:
            print(e)
    data = None
    if path is not None:
        data = loadmat(path)
        data = remove_keys(data, '__')
        for key in data.keys():
            data[key] = reduce_dimensions(data[key])
            data[key] = convert_to_list(data[key])
            if data[key] == '' or data[key] == []:
                data[key] = None
            if key == 'ch_names':
                data[key] = [name.replace(' ', '') for name in data[key]]
            try:
                if key == 'events' and data[key] is not None:
                    for i, event, duration, label in enumerate(data[key]):
                        if isinstance(event, str):
                            event = float(event.replace(' ', ''))
                        if isinstance(duration, str):
                            duration = float(duration.replace(' ', ''))
                        if isinstance(label, str):
                            label = label.replace(' ', '')
                        data[key][i] = [event, duration, label]
            except:
                pass
    return data


def read_edf(path=None):
    """
        读取欧洲数据格式(.edf)文件。

        Parameters
        ----------
        path : str, optional
            .edf文件路径。如果为None，会打开文件选择对话框

        Returns
        -------
        dict
            EEG数据结构：

            - **data**: 各通道时间序列（数字信号转为微伏）
            - **srate**: 采样率(Hz)
            - **events**: 标注事件([起始时间, 时长, 描述])
            - **nchan**: 通道数量
            - **ch_names**: 通道名称（去除'.'字符）
            - **units**: 物理单位
            - **type**: 固定为'eeg'
            - **montage**: 电极位置占位符

        Notes
        -----
        数字值通过乘以1e-6转为电压值。
        通道名称去除'.'字符确保兼容性。

        Examples:
        ------
        >>> edf_data = read_edf("eeg_record.edf")
        >>> print(f"采样率: {edf_data['srate']} Hz")
        采样率: 250.0 Hz
        >>> # 绘制第一个通道的数据
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(edf_data['data'][0][:1000])
        >>> plt.title(edf_data['ch_names'][0])
        """
    if path is None:
        root = Tk()
        root.withdraw()
        try:
            # select bdf or edf file
            path = filedialog.askopenfilenames(initialdir='/', title='Select one edf files',
                                               filetypes=(("one edf file", "*.edf"),))[0]
        except Exception as e:
            print(e)
    data = None
    if path is not None:
        edf_file = pyedflib.EdfReader(path)
        edf_data = []
        nchan = edf_file.signals_in_file
        for chan in range(nchan):
            edf_data.append(edf_file.readSignal(chan, digital=True) * 0.000001)
        data_dict = {}
        data_dict['data'] = edf_data
        data_dict['srate'] = edf_file.getSampleFrequencies()[0]
        data_dict['events'] = edf_file.readAnnotations()
        data_dict['nchan'] = edf_file.signals_in_file
        data_dict['ch_names'] = [chan.replace('.', '') for chan in edf_file.getSignalLabels()]
        data_dict['units'] = [edf_file.getPhysicalDimension(i) for i in range(nchan)]
        data_dict['type'] = 'eeg'
        data_dict['montage'] = None
        data = data_dict
        edf_file.close()
    return data


def read_bdf(path=None, type='eeg', montage=None):
    """
        读取Biosemi数据格式(.bdf)EEG文件。

        Parameters
        ----------
        path : str, optional
            .bdf文件路径。如果为None，会打开文件选择对话框
        type : str, default='eeg'
            数据类型标识
        montage : any, optional
            电极布局信息（当前未使用）

        Returns
        -------
        dict
            EEG数据结构：

            - **data**: 各通道时间序列（数字信号转为电压）
            - **srate**: 采样率(Hz)
            - **events**: 处理后的事件标记[[时间, 时长, 标签]]
            - **nchan**: 通道数量
            - **ch_names**: 通道名称（去除特殊字符）
            - **units**: 物理单位
            - **type**: 指定的数据类型
            - **montage**: 输入的电极布局占位符
            - **file_info**: 文件头信息字典

        Notes
        -----
        1. 数字信号转为电压值（×1e-6）
        2. 事件标签转换为整数标签
        3. 通道名称去除特殊字符

        Examples:
        ------
        >>> bdf_data = read_bdf("biosemi_data.bdf")
        >>> print(f"共 {bdf_data['nchan']} 个通道")
        共 64 个通道
        >>> # 查看文件信息
        >>> print(f"数字最大值: {bdf_data['file_info']['DigitalMaximum']}")
        数字最大值: 8388607
        """
    if path is None:
        root = Tk()
        root.withdraw()
        try:
            # select bdf or edf file
            path = filedialog.askopenfilenames(initialdir='/', title='Select one bdf files',
                                               filetypes=(("one bdf file", "*.bdf"),))[0]
        except Exception as e:
            print(e)
    data = None
    if path is not None:
        bdf_file = pyedflib.EdfReader(path)
        bdf_data = []
        nchan = bdf_file.signals_in_file
        for chan in range(nchan):
            bdf_data.append(bdf_file.readSignal(chan, digital=False) * 0.000001)
        data_dict = {}
        data_dict['data'] = bdf_data
        data_dict['srate'] = bdf_file.getSampleFrequencies()[0]
        data_dict['events'] = np.array(bdf_file.readAnnotations()).T.tolist()
        for idx, event in enumerate(data_dict['events']):
            time, duration, label = event
            label = str(int(float(label)))
            time = float(time)
            duration = float(duration)
            data_dict['events'][idx] = [time, duration, label]
        data_dict['nchan'] = bdf_file.signals_in_file
        data_dict['ch_names'] = [chan.replace('.', '') for chan in bdf_file.getSignalLabels()]
        data_dict['units'] = [bdf_file.getPhysicalDimension(i) for i in range(nchan)]
        data_dict['type'] = type
        data_dict['montage'] = montage
        data_dict['file_info'] = {'PhysicalMaximum': bdf_file.getPhysicalMaximum(),
                                  'PhysicalMinimum': bdf_file.getPhysicalMinimum(),
                                  'DigitalMaximum': bdf_file.getDigitalMaximum(),
                                  'DigitalMinimum': bdf_file.getDigitalMinimum()}
        bdf_file.close()
        data = data_dict.copy()
    return data


def read_snirf(path=None, type='fnirs'):
    """
        读取SNIRF格式(.snirf)近红外光谱数据。

        Parameters
        ----------
        path : str, optional
            .snirf文件路径。如果为None，会打开文件选择对话框
        type : str, default='fnirs'
            数据类型标识

        Returns
        -------
        dict
            fNIRS数据结构：

            - **data**: 时间序列数据（不同波长分离）
            - **time**: 时间向量
            - **srate**: 计算得出的采样率
            - **events**: 所有刺激通道的事件标记
            - **nchan**: 通道数量
            - **ch_names**: 通道名称
            - **type**: 指定数据类型
            - **loc**: 源/探测器3D位置
            - **sd**: 源-探测器配对索引
            - **wavelengths**: 测量波长

        Notes
        -----
        封装read_minilab_snirf，整合多通道事件标记。

        Examples:
        ------
        >>> fnirs_data = read_snirf("fnirs_data.snirf")
        >>> print(f"源-探测器配对: {fnirs_data['sd'][:3]}")
        源-探测器配对: [(1, 1), (1, 2), (1, 3)]
        >>> # 使用MNE进一步处理[8,9](@ref):
        >>> import mne
        >>> raw = mne.io.RawArray(fnirs_data['data'].T,
        >>>                      mne.create_info(fnirs_data['ch_names'],
        >>>                      fnirs_data['srate'],
        >>>                      ch_types='fnirs_cw_amplitude'))
        """
    if path is None:
        root = Tk()
        root.withdraw()
        try:
            # select bdf or edf file
            path = filedialog.askopenfilenames(initialdir='/', title='Select one snirf files',
                                               filetypes=(("one snirf file", "*.snirf"),))[0]
        except Exception as e:
            print(e)
    data = None
    if path is not None:
        data = read_minilab_snirf(path, type=type)
    return data


def read_json(path=None):
    """
        读取JSON格式元数据文件。

        Parameters
        ----------
        path : str, optional
            .json文件路径。如果为None，会打开文件选择对话框

        Returns
        -------
        dict or None
            解析后的JSON内容字典。
            如文件不存在返回None。
        """
    if path is None:
        root = Tk()
        root.withdraw()
        try:
            # select bdf or edf file
            path = filedialog.askopenfilenames(initialdir='/', title='Select one json files',
                                               filetypes=(("one json file", "*.json"),))[0]
        except Exception as e:
            print(e)
    data = None
    if path is not None:
        with open(path, 'r') as json_file:
            data = json.load(json_file)
    return data


def read_minilab_snirf(path=None, type='fnirs'):
    """
        解析MiniLab兼容的SNIRF fNIRS数据结构。

        Parameters
        ----------
        path : str
            .snirf文件路径
        type : str, default='fnirs'
            数据类型标识

        Returns
        -------
        dict
            结构化fNIRS数据，包含以下键：

            - **data**: 时间序列数据（2D数组）
            - **time**: 时间向量（1D数组）
            - **srate**: 采样率（根据时间间隔计算）
            - **events**: 所有刺激通道的事件
            - **nchan**: 通道数量
            - **ch_names**: 探头标记名称
            - **sd**: 从通道名解析的源-探测器配对
            - **loc**: 源/探测器/标记位置的3D坐标
            - **wavelengths**: 测量波长

        Notes
        -----
        1. 处理多刺激通道事件标记
        2. 根据时间向量计算采样率
        3. 从通道名称解析源-探测器索引
        4. 返回整合的3D位置数据

        Examples:
        ------
        >>> fnirs = read_minilab_snirf("task_motor.snirf")
        >>> # 查看3D坐标信息
        >>> print(f"第一个光源位置: {fnirs['loc']['sourcePos3D'][0]}")
        第一个光源位置: [45.2, 32.1, 85.3]
        >>> # 计算平均血红蛋白浓度[9](@ref):
        >>> hbo = fnirs['data'][:, ::2].mean(axis=1)  # 奇数索引为HbO
        >>> plt.plot(fnirs['time'], hbo)
        >>> plt.xlabel("时间(s)")
        >>> plt.ylabel("HbO (μM)")
        """
    snirf_file = snirf.loadSnirf(path)
    data_dict = {}
    data_dict['data'] = snirf_file.nirs[0].data[0].dataTimeSeries
    data_dict['time'] = snirf_file.nirs[0].data[0].time
    data_dict['srate'] = 1.0 / np.mean(np.diff(data_dict['time']))  # 采样率从时间间隔中计算
    if len(snirf_file.nirs[0].stim) > 0:
        # 创建一个列表来存储所有的事件
        all_events = []
        for stim in snirf_file.nirs[0].stim:
            # 检查 stim.data 是否存在且非空
            if stim.data is not None and len(stim.data) > 0:
                # 遍历当前 stim 的所有事件
                for event in stim.data:
                    time, duration, label = event
                    label = str(int(float(label)))  # 确保 label 为字符串
                    time = float(time)
                    duration = float(duration)
                    all_events.append([time, duration, label])

        # 对所有事件按照 time 排序
        all_events.sort(key=lambda x: x[0])

        # 如果有事件，保存到字典；否则设置为 None
        data_dict['events'] = all_events if all_events else None
    else:
        data_dict['events'] = None

    data_dict['nchan'] = snirf_file.nirs[0].data[0].dataTimeSeries.shape[1]
    data_dict['ch_names'] = snirf_file.nirs[0].probe.landmarkLabels
    data_dict['type'] = type
    loc = {
        'sourcePos3D': snirf_file.nirs[0].probe.sourcePos3D,
        'detectorPos3D': snirf_file.nirs[0].probe.detectorPos3D,
        'landmarkPos3D': snirf_file.nirs[0].probe.landmarkPos3D
    }
    data_dict['loc'] = loc

    sd_pairs = []
    for ch_name in data_dict['ch_names'][:data_dict['nchan'] // 2]:
        sd_part = ch_name.split(" ")[0]
        source, detector = sd_part.split("_")
        source_idx = int(source[1:])
        detector_idx = int(detector[1:])
        sd_pairs.append((source_idx, detector_idx))
    data_dict['sd'] = sd_pairs
    data_dict['wavelengths'] = snirf_file.nirs[0].probe.wavelengths

    return data_dict


def save_file(data, save_path=None, save_filestyle='mat'):
    """
    Saves data to a file in specified format

    :param data: Data to save
    :type data: dict
    :param save_path: Path to save file, defaults to None (opens dialog)
    :type save_path: str, optional
    :param save_filestyle: File format ('mat', 'csv', 'json', 'txt'), defaults to 'mat'
    :type save_filestyle: str, optional
    """
    if save_path:
        if save_filestyle == 'mat':
            save_mat(data, save_path)
        elif save_filestyle == 'json':
            save_json(data, save_path)
    else:
        if save_filestyle == 'mat':
            save_mat(data)
        elif save_filestyle == 'json':
            save_json(data)

def save_json(data, path=None):
    """
    Saves data to a JSON file

    :param data: Data to save
    :type data: dict
    :param path: Save path, defaults to None (opens dialog)
    :type path: str, optional
    :raises ValueError: If data validation fails
    """
    root = Tk()
    root.withdraw()
    if path is None:
        try:
            path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("json file", "*.json")],
                                                 title="Select file save path")
        except Exception as e:
            print(e)
    if path:
        if check_data_dict(data):
            data = save_prepare(data)
            with open(path, 'w') as json_file:
                json.dump(data, json_file, indent=2)

def check_data_dict(data_dict):
    """
    Validates the structure of a data dictionary

    :param data_dict: Data dictionary to validate
    :type data_dict: dict
    :return: True if valid
    :rtype: bool
    :raises TypeError: If input is not a dictionary
    :raises ValueError: If required fields are missing or invalid
    """
    if isinstance(data_dict, dict):
        if isinstance(data_dict['data'], np.ndarray) or isinstance(data_dict['data'], list):
            if data_dict['srate']:
                return True
            else:
                raise ValueError("srate should not be None")
        else:
            raise ValueError("data should not be None")
    else:
        raise TypeError("data_dict should be a dict")


def save_mat(data, path=None):
    """
    Saves data to a MATLAB .mat file

    :param data: Data to save
    :type data: dict
    :param path: Save path, defaults to None (opens dialog)
    :type path: str, optional
    """
    root = Tk()
    root.withdraw()
    if path is None:
        try:
            path = filedialog.asksaveasfilename(defaultextension=".mat", filetypes=[("mat files", "*.mat")],
                                                title="Select file save path")
        except Exception as e:
            print(e)
    if path:
        data = save_prepare(data)
        savemat(path, data)


def save_prepare(data):
    """
    Prepares data for saving by converting to lists and handling empty values

    :param data: Data to prepare
    :type data: dict
    :return: Prepared data
    :rtype: dict
    """

    def convert_to_list(data):
        """
        Converts numpy arrays to Python lists recursively

        :param data: Input data (array or nested structure)
        :type data: any
        :return: Converted data
        :rtype: list or primitive
        """
        if isinstance(data, float):
            return data
        if isinstance(data, (np.ndarray, np.generic)):
            if data.ndim == 0:
                return data
            if data.dtype.kind in ('S', 'U'):
                return [x.decode('utf-8') if isinstance(x, bytes) else x for x in data.flat]
            return [convert_to_list(item) for item in data]
        if isinstance(data, (list, tuple)):
            return [convert_to_list(item) for item in data]
        return data

    for key in data.keys():
        if isinstance(data[key], np.ndarray):
            data[key] = convert_to_list(data[key])
        elif isinstance(data[key], frozenset):
            data[key] = convert_to_list(data[key])
    data = {key: '' if value is None else value for key, value in data.items()}
    return data