#%%

from mne.datasets import misc
from mne.io import read_raw_eyelink,read_raw_edf
import os
import os.path as op

#%%
##.asc
examples_dir = "./osfstorage-archive/eye tracking raw data/ASC"      
edf_file = op.join(examples_dir, 'AE01S.asc')    
raw = read_raw_eyelink(edf_file)
custom_scalings = dict(eyegaze=1e3)
raw.pick(picks="eyetrack").plot(scalings=custom_scalings)
# %%
#.edf
import mne

# 定义 .edf 文件的路径
edf_file_path = './osfstorage-archive/eye tracking raw data/_edf-Files/AE01S.edf'

# 使用 mne.io.read_raw_edf 函数读取 .edf 文件
raw_edf = mne.io.read_raw_edf(edf_file_path, preload=True)

# 查看数据的一些基本信息
print(raw_edf.info)

# 绘制原始数据
raw_edf.plot()


# %%
