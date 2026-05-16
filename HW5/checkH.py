import torch

ckpt_path = r"E:\JupyterPjs\AI4Chem\HW5\pretrained\checkpoints\500000.pt"

# 加上 weights_only=False 强行解包
checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

# GeoDiff 的模型权重通常在 'model' 这个 key 里
state_dict = checkpoint.get('model', checkpoint)

for key, tensor in state_dict.items():
    if 'embedding' in key or 'atom_emb' in key or 'node_emb' in key:
        print(f"🔍 找到原子嵌入层: {key}")
        print(f"📏 该层的维度是: {tensor.shape}")