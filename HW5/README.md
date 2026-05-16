# 如何装环境：
1. conda create -n geodiff_modified python=3.10 -y
2. conda activate geodiff_modified
3. conda install -c conda-forge rdkit=2023.09.5 -y
4. pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
5. pip install torch_geometric
6. pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
7. pip install numpy==1.26.4 scipy pandas scikit-learn networkx h5py matplotlib seaborn tensorboard tqdm jupyterlab ipython pydantic pyyaml requests cryptography easydict

# 如何训练：
1. 检查配置文件，并确保你的conda环境是geodiff_modified
2. python -u main.py --mode train --config configs/qm9_default.yml --device cuda:0 (如果你用cpu的话这里就是cpu，但是cpu会特别慢) --workdir logs

# 如何采样：
1. python -u main.py --mode sample --device cuda:0 --ckpt log/xxx/checkpoints/xxx.pt (请根据实际情况修改) --start_idx 0 --end_idx 20000
2. 注意：采样的时候用的是log目录下的config文件，而不是训练时用的，所以要改配置文件就去checkpoints目录同级的目录下修改对应的config文件

# 如何创建自己的数据集&从哪里下载训练用的数据集
1. 准备好包含mol文件和arc文件的目录（需要文件名一一对应），然后python make_data.py，按照选项设置
2. 如何下载训练数据集：
    1. 百度网盘：https://pan.baidu.com/s/1pymnx8wSoiAEkmlbyCgXxg 提取码: ycwb 
    2. Google drive: https://drive.google.com/file/d/1963L6ZNjHb867Qpo4D94GDySryQD_uoj/view?usp=sharing