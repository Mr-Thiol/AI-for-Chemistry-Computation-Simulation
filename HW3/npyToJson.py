import numpy as np
import json

def npy_to_json(npy_file, json_file):
    # 读取 npy (注意：allow_pickle=True 且加上 .item() 将对象转回字典)
    try:
        features = np.load(npy_file, allow_pickle=True).item()
    except FileNotFoundError:
        print(f"找不到文件 {npy_file}，请检查路径。")
        return

    json_data = {}
    
    # 遍历字典，将 numpy array 转化为标准 python list，并保留小数位
    for idx, fp in features.items():
        # int(idx) 是为了保证 JSON 的键是标准格式
        # round(val, 4) 让你的作业报告不会被又长又臭的浮点数撑爆
        json_data[int(idx)] = [round(float(val), 4) for val in fp]

    # 写入 JSON
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=4)
        
    print(f"✅ 完美转换！数据已从 {npy_file} 提取并保存至 {json_file}。")
    print("你可以直接用记事本或浏览器打开这个 JSON 文件进行截图了。")

if __name__ == "__main__":
    npy_to_json('Ti_ACSF_features_Modular.npy', 'Ti_ACSF_features.json')