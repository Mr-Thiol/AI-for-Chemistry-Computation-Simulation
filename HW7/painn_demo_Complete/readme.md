### 训练 
```
1.准备数据 ./data/train/raw force.arc structure.arc
2.处理数据 python load_data.py
3.python main.py # 训练
```

## 环境准备
### 安装python
https://www.python.org/downloads/
### 安装torch
```
# cuda驱动版本12.4
# pytorch 从2.3以后开始支持numpy2.0
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia
# pyg
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
```
