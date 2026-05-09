import pickle
import json
import numpy as np

def convert_numpy(obj):
    """
    递归将数据中的 NumPy 对象转换为标准 Python 类型，以便进行 JSON 序列化。
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # 数组转为列表
    elif isinstance(obj, np.generic):
        return obj.item()    # NumPy 标量转为 Python 标量 (int, float 等)
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    return obj

def pkl_to_json(input_file, output_file):
    try:
        # 1. 以二进制读取模式打开 pkl 文件
        print(f"正在读取 {input_file}...")
        with open(input_file, 'rb') as f:
            data = pickle.load(f)

        # 2. 转换数据格式（处理 NumPy 类型）
        print("正在转换数据格式...")
        json_ready_data = convert_numpy(data)

        # 3. 写入 JSON 文件
        print(f"正在保存到 {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            # indent=4 让生成的 JSON 文件具有缩进，方便人类阅读
            # ensure_ascii=False 确保中文或特殊符号能正常显示
            json.dump(json_ready_data, f, indent=4, ensure_ascii=False)
        
        print("转换成功！")

    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    # 请确保该脚本与 neighbor_data.pkl 放在同一目录下，或填写完整路径
    pkl_to_json('neighbor_data.pkl', 'neighbor_data.json')