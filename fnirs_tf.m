%% ======================== 用户设置 =========================
root_folder = '...SEFMID\Raw data\fNIRS'; %选择原始文件路径
save_root   = '...SEFMID\Raw data\fNIRS_snirf'; %选择格式转换保存路径

if ~exist(save_root, 'dir')
    mkdir(save_root);
end

addpath('E:\MATLAB R2022b(64bit)\toolbox\nirs-toolbox-master\nirs-toolbox-master'); %需安装nirs-toolbox工具包https://github.com/huppertt/nirs-toolbox

% 选择模式
% mode = 1 : 遍历所有子文件夹
% mode = 2 : 手动选择指定子文件夹
mode = 1;

% 如果选择手动模式，指定子文件夹名称
selected_folders = {'24001'};  % 例子，可修改

%% ======================== 获取文件夹列表 =========================
folder_list = dir(root_folder);
folder_list = folder_list([folder_list.isdir]);
folder_list = folder_list(~ismember({folder_list.name},{'.','..'}));

switch mode
    case 1
        % 遍历所有子文件夹
        folders_to_process = {folder_list.name};
    case 2
        % 使用用户指定的文件夹
        folders_to_process = selected_folders;
    otherwise
        error('mode 设置错误，请选择 1 或 2');
end

%% ======================== 遍历处理 =========================
for i = 1:length(folders_to_process)
    subject_id     = folders_to_process{i};
    subject_folder = fullfile(root_folder, subject_id);
    
    if ~exist(subject_folder, 'dir')
        fprintf('文件夹 %s 不存在，跳过\n', subject_folder);
        continue;
    end
    
    try
        % 加载该被试的所有fNIRS数据
        raw = nirs.io.loadDirectory(subject_folder);

        % 保存路径（直接保存为 24001.snirf）
        save_file = fullfile(save_root, [subject_id '.snirf']);

        % 保存第一个 raw 数据（如有多个文件，可改进保存多个）
        nirs.io.saveSNIRF(raw(1), save_file);

        fprintf('成功保存 %s\n', save_file);

    catch ME
        fprintf('处理 %s 时出错：%s\n', subject_id, ME.message);
    end
end

fprintf('\n所有被试处理完成。\n');
