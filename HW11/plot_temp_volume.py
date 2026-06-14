import numpy as np
import matplotlib.pyplot as plt
import os

# 设置字体
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

def parse_lasp_output(filepath):
    """
    解析LASP输出文件，提取温度和体积数据
    """
    temperatures = []
    volumes = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip().startswith('istep'):
                parts = line.split()
                if len(parts) >= 11:
                    try:
                        # 温度在索引3，体积在索引10
                        temp = float(parts[3])
                        vol = float(parts[10])
                        temperatures.append(temp)
                        volumes.append(vol)
                    except (ValueError, IndexError):
                        continue
    
    return np.array(temperatures), np.array(volumes)

def main():
    # 定义四个文件夹路径
    folders = [
        'MD-LASP-Al216_0',
        'MD-LASP-Al216_1',
        'MD-LASP-Al216_2',
        'MD-LASP-Al216_3'
    ]
    
    # 存储所有数据
    all_temps = []
    all_vols = []
    all_labels = []
    
    # 读取三个文件
    for folder in folders:
        filepath = os.path.join(folder, 'lasp.out')
        if os.path.exists(filepath):
            temps, vols = parse_lasp_output(filepath)
            all_temps.append(temps)
            all_vols.append(vols)
            all_labels.append(folder)
            print(f'{folder}: 读取了 {len(temps)} 个数据点')
            print(f'  温度范围: {temps.min():.1f} - {temps.max():.1f} K')
            print(f'  体积范围: {vols.min():.1f} - {vols.max():.1f} Å³')
        else:
            print(f'警告: 未找到文件 {filepath}')
    
    # 合并所有数据
    all_temps_combined = np.concatenate(all_temps)
    all_vols_combined = np.concatenate(all_vols)
    
    # 找到整体的最高温度点
    max_temp_global = np.max(all_temps_combined)
    max_temp_idx_global = np.argmax(all_temps_combined)
    
    print(f'\n全局最高温度: {max_temp_global:.1f} K')
    
    # 计算每个文件的平均温度
    avg_temps = [np.mean(temps) for temps in all_temps]
    print(f'\n各文件平均温度:')
    for label, avg_temp in zip(all_labels, avg_temps):
        print(f'  {label}: {avg_temp:.1f} K')
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 用于记录是否已添加图例
    heating_label_added = False
    cooling_label_added = False
    
    # 用于存储升温和降温的数据
    heating_temps = []
    heating_vols = []
    cooling_temps = []
    cooling_vols = []
    
    # 逐个文件绘制，使用滑动窗口判断升温/降温趋势
    cumulative_idx = 0
    for i, (temps, vols, label) in enumerate(zip(all_temps, all_vols, all_labels)):
        start_idx = cumulative_idx
        end_idx = cumulative_idx + len(temps)
        
        # 找到这个文件中温度的最高点
        local_max_idx = np.argmax(temps)
        local_max_temp = temps[local_max_idx]
        
        print(f'\n{label}:')
        print(f'  温度范围: {temps.min():.1f} - {temps.max():.1f} K')
        print(f'  局部最高温度: {local_max_temp:.1f} K (在第 {local_max_idx} 个点)')
        
        # 策略：在到达最高温度之前是升温，之后是降温
        # 但要考虑整体的温度序列
        
        # 简单方法：如果这个文件的数据点在全局最高温之前，则升温；之后则降温
        if end_idx <= max_temp_idx_global:
            # 整个文件都在全局最高温之前，全部标记为升温
            color = 'red'
            label_text = 'Heating' if not heating_label_added else ''
            heating_label_added = True
            plt.scatter(temps, vols, c=color, alpha=0.3, s=35,
                       label=label_text if label_text else None)
            heating_temps.extend(temps)
            heating_vols.extend(vols)
            print(f'  阶段: 升温')
        elif start_idx > max_temp_idx_global:
            # 整个文件都在全局最高温之后，全部标记为降温
            color = 'blue'
            label_text = 'Cooling' if not cooling_label_added else ''
            cooling_label_added = True
            plt.scatter(temps, vols, c=color, alpha=0.3, s=35,
                       label=label_text if label_text else None)
            cooling_temps.extend(temps)
            cooling_vols.extend(vols)
            print(f'  阶段: 降温')
        else:
            # 这个文件包含全局最高温点，需要分段绘制
            split_idx = max_temp_idx_global - start_idx
            
            # 升温部分（包括最高点）
            if not heating_label_added:
                plt.scatter(temps[:split_idx+1], vols[:split_idx+1], c='red', alpha=0.3, s=35,
                           label='Heating')
                heating_label_added = True
            else:
                plt.scatter(temps[:split_idx+1], vols[:split_idx+1], c='red', alpha=0.3, s=35)
            heating_temps.extend(temps[:split_idx+1])
            heating_vols.extend(vols[:split_idx+1])
            
            # 降温部分
            if split_idx + 1 < len(temps):
                if not cooling_label_added:
                    plt.scatter(temps[split_idx+1:], vols[split_idx+1:], c='blue', alpha=0.3, s=35,
                               label='Cooling')
                    cooling_label_added = True
                else:
                    plt.scatter(temps[split_idx+1:], vols[split_idx+1:], c='blue', alpha=0.3, s=35)
                cooling_temps.extend(temps[split_idx+1:])
                cooling_vols.extend(vols[split_idx+1:])
            
            print(f'  阶段: 升温({split_idx+1}点) + 降温({len(temps)-split_idx-1}点)')
        
        cumulative_idx = end_idx
    
    # 转换为numpy数组
    heating_temps = np.array(heating_temps)
    heating_vols = np.array(heating_vols)
    cooling_temps = np.array(cooling_temps)
    cooling_vols = np.array(cooling_vols)
    
    # 定义分箱平滑函数
    def bin_smooth(temps, vols, n_bins=50):
        """
        对温度-体积数据进行分箱平滑
        返回: 分箱后的温度、平均体积
        """
        # 排序
        sort_idx = np.argsort(temps)
        temps_sorted = temps[sort_idx]
        vols_sorted = vols[sort_idx]
        
        # 分箱
        bins = np.linspace(temps_sorted.min(), temps_sorted.max(), n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_vols = []
        
        for i in range(n_bins):
            mask = (temps_sorted >= bins[i]) & (temps_sorted < bins[i+1])
            if np.sum(mask) > 0:
                bin_vols.append(np.mean(vols_sorted[mask]))
            else:
                bin_vols.append(np.nan)
        
        # 移除NaN值
        bin_vols = np.array(bin_vols)
        valid_mask = ~np.isnan(bin_vols)
        
        return bin_centers[valid_mask], bin_vols[valid_mask]
    
    def gaussian_smooth(data, sigma=2.0):
        """
        高斯平滑（二次降噪）- 使用镜像填充避免边缘效应
        sigma: 高斯核的标准差，越大越平滑
        """
        # 创建高斯核
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # 生成高斯核
        x = np.arange(kernel_size) - kernel_size // 2
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / kernel.sum()
        
        # 对数据边缘进行镜像填充
        pad_size = kernel_size // 2
        # 左边镜像
        left_pad = data[1:pad_size+1][::-1]
        # 右边镜像
        right_pad = data[-(pad_size+1):-1][::-1]
        # 拼接
        padded_data = np.concatenate([left_pad, data, right_pad])
        
        # 对填充后的数据进行卷积
        smoothed_padded = np.convolve(padded_data, kernel, mode='same')
        
        # 裁剪掉填充部分，返回原始长度
        smoothed = smoothed_padded[pad_size:-pad_size]
        
        return smoothed
    
    def find_transition_with_edge_clip(temps, dV_dT, find_max=True, edge_clip=0.15):
        """
        在限定区间内寻找相变点（边缘裁切）
        edge_clip: 裁切掉头尾各多少比例的数据（默认15%）
        find_max: True寻找最大值（升温），False寻找最小值（降温）
        """
        n = len(temps)
        # 计算有效搜索范围
        start_idx = int(n * edge_clip)
        end_idx = int(n * (1 - edge_clip))
        
        # 确保至少有10个点
        if end_idx - start_idx < 10:
            start_idx = 0
            end_idx = n
            print(f'  警告: 数据点太少，跳过边缘裁切')
        else:
            print(f'  边缘裁切: 保留 {temps[start_idx]:.1f}K - {temps[end_idx-1]:.1f}K 区间')
        
        # 在有效范围内寻找极值
        if find_max:
            relative_idx = np.argmax(dV_dT[start_idx:end_idx])
        else:
            relative_idx = np.argmin(dV_dT[start_idx:end_idx])
        
        # 转换为绝对索引
        abs_idx = start_idx + relative_idx
        
        return abs_idx
    
    # 分析升温阶段
    heating_transition_temp = None
    heating_transition_vol = None
    if len(heating_temps) > 10:
        try:
            # 排序数据
            sort_idx = np.argsort(heating_temps)
            heating_temps_sorted = heating_temps[sort_idx]
            heating_vols_sorted = heating_vols[sort_idx]
            
            # 分箱平滑（第一次）
            bin_temps_h, bin_vols_h = bin_smooth(heating_temps_sorted, heating_vols_sorted, n_bins=50)
            
            # 高斯平滑（第二次降噪）
            bin_vols_h_smooth = gaussian_smooth(bin_vols_h, sigma=2.0)
            print(f'  应用高斯平滑 (sigma=2.0)')
            
            # 计算一阶差分 dV/dT
            dV_dT = np.diff(bin_vols_h_smooth) / np.diff(bin_temps_h)
            temps_deriv = (bin_temps_h[:-1] + bin_temps_h[1:]) / 2
            
            # 在限定区间内找到dV/dT最大的点（体积膨胀率最大）
            max_slope_idx = find_transition_with_edge_clip(temps_deriv, dV_dT, find_max=True, edge_clip=0.15)
            heating_transition_temp = temps_deriv[max_slope_idx]
            
            # 找到对应的体积（在原始数据中插值）
            heating_transition_vol = np.interp(heating_transition_temp, bin_temps_h, bin_vols_h_smooth)
            
            # 绘制平滑曲线（显示平滑后的数据）
            plt.plot(bin_temps_h, bin_vols_h_smooth, 'r-', linewidth=2.5, 
                    label='Heating (binned + smoothed)', alpha=0.9)
            
            print(f'\n=== 升温阶段分析 ===')
            print(f'数据点数: {len(heating_temps_sorted)}')
            print(f'分箱数: {len(bin_temps_h)}')
            print(f'最大体积膨胀率 dV/dT: {dV_dT[max_slope_idx]:.6f} Ų/K')
            print(f'升温突变点 T⁺: {heating_transition_temp:.2f} K')
            print(f'对应体积: {heating_transition_vol:.2f} Ų')
            
        except Exception as e:
            print(f'\n升温阶段分析失败: {e}')
    
    # 分析降温阶段
    cooling_transition_temp = None
    cooling_transition_vol = None
    if len(cooling_temps) > 10:
        try:
            print(f'\n=== 降温阶段分析 ===')
            
            # 排序数据
            sort_idx = np.argsort(cooling_temps)
            cooling_temps_sorted = cooling_temps[sort_idx]
            cooling_vols_sorted = cooling_vols[sort_idx]
            
            # 分箱平滑（第一次）
            bin_temps_c, bin_vols_c = bin_smooth(cooling_temps_sorted, cooling_vols_sorted, n_bins=50)
            
            # 高斯平滑（第二次降噪）
            bin_vols_c_smooth = gaussian_smooth(bin_vols_c, sigma=2.0)
            print(f'  应用高斯平滑 (sigma=2.0)')
            
            # 计算一阶差分 dV/dT
            dV_dT = np.diff(bin_vols_c_smooth) / np.diff(bin_temps_c)
            temps_deriv = (bin_temps_c[:-1] + bin_temps_c[1:]) / 2
            
            # 在限定区间内找到dV/dT最小的点（体积收缩率最大，为负值）
            min_slope_idx = find_transition_with_edge_clip(temps_deriv, dV_dT, find_max=False, edge_clip=0.15)
            cooling_transition_temp = temps_deriv[min_slope_idx]
            
            # 找到对应的体积（在原始数据中插值）
            cooling_transition_vol = np.interp(cooling_transition_temp, bin_temps_c, bin_vols_c_smooth)
            
            # 绘制平滑曲线（显示平滑后的数据）
            plt.plot(bin_temps_c, bin_vols_c_smooth, 'b-', linewidth=2.5,
                    label='Cooling (binned + smoothed)', alpha=0.9)
            
            print(f'\n--- 降温相变分析 ---')
            print(f'数据点数: {len(cooling_temps_sorted)}')
            print(f'分箱数: {len(bin_temps_c)}')
            print(f'最大体积收缩率 dV/dT: {dV_dT[min_slope_idx]:.6f} Ų/K')
            print(f'降温突变点 T⁻: {cooling_transition_temp:.2f} K')
            print(f'对应体积: {cooling_transition_vol:.2f} Ų')
            
            # # 在500K之后寻找dV/dT最大的点（T-'）- 已注释
            # print(f'\n--- 500K后二次相变分析 ---')
            # mask_500K = temps_deriv >= 500
            # if np.sum(mask_500K) > 10:
            #     temps_after_500 = temps_deriv[mask_500K]
            #     dV_dT_after_500 = dV_dT[mask_500K]
            #     
            #     # 找到dV/dT最大的点
            #     max_slope_idx_after_500 = np.argmax(dV_dT_after_500)
            #     cooling_transition_temp_2 = temps_after_500[max_slope_idx_after_500]
            #     cooling_transition_vol_2 = np.interp(cooling_transition_temp_2, bin_temps_c, bin_vols_c_smooth)
            #     
            #     print(f'搜索区间: ≥500K')
            #     print(f'有效数据点: {len(temps_after_500)}')
            #     print(f'最大体积膨胀率 dV/dT: {dV_dT_after_500[max_slope_idx_after_500]:.6f} Ų/K')
            #     print(f'降温二次突变点 T⁻\': {cooling_transition_temp_2:.2f} K')
            #     print(f'对应体积: {cooling_transition_vol_2:.2f} Ų')
            # else:
            #     cooling_transition_temp_2 = None
            #     cooling_transition_vol_2 = None
            #     print(f'500K后数据点太少，无法分析')
            cooling_transition_temp_2 = None
            cooling_transition_vol_2 = None
            
        except Exception as e:
            print(f'\n降温阶段分析失败: {e}')
            cooling_transition_temp_2 = None
            cooling_transition_vol_2 = None
    else:
        cooling_transition_temp_2 = None
        cooling_transition_vol_2 = None
    
    # 在图上标记相变点
    print(f'\n=== 相变温度总结 ===')
    if heating_transition_temp is not None:
        plt.scatter([heating_transition_temp], [heating_transition_vol], c='orangered', s=200, 
                   marker='*', alpha=1.0, edgecolors='darkred', linewidth=2,
                   label=f'T+ (Melting): {heating_transition_temp:.1f} K', zorder=5)
        
        # 绘制垂直辅助线
        plt.axvline(x=heating_transition_temp, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
        
        print(f'升温突变点 T+ (熔化点): {heating_transition_temp:.2f} K')
        
    if cooling_transition_temp is not None:
        plt.scatter([cooling_transition_temp], [cooling_transition_vol], c='deepskyblue', s=200, 
                   marker='*', alpha=1.0, edgecolors='darkblue', linewidth=2,
                   label=f'T- (Solidification): {cooling_transition_temp:.1f} K', zorder=5)
        
        # 绘制垂直辅助线
        plt.axvline(x=cooling_transition_temp, color='blue', linestyle=':', linewidth=1.5, alpha=0.5)
        
        print(f'降温突变点 T- (凝固点): {cooling_transition_temp:.2f} K')
    
    # # 紫色点标注 - 已注释
    # if cooling_transition_temp_2 is not None:
    #     plt.scatter([cooling_transition_temp_2], [cooling_transition_vol_2], c='purple', s=200, 
    #                marker='D', alpha=1.0, edgecolors='darkviolet', linewidth=2,
    #                label=f'T-\' (2nd transition): {cooling_transition_temp_2:.1f} K', zorder=5)
    #     
    #     # 绘制垂直辅助线
    #     plt.axvline(x=cooling_transition_temp_2, color='purple', linestyle=':', linewidth=1.5, alpha=0.5)
    #     
    #     print(f'降温二次突变点 T-\' (500K后): {cooling_transition_temp_2:.2f} K')
    
    # # 过冷度计算 - 已注释
    # if heating_transition_temp is not None and cooling_transition_temp is not None:
    #     supercooling = heating_transition_temp - cooling_transition_temp
    #     print(f'\n过冷度 (Supercooling): {supercooling:.2f} K')
    #     print(f'过冷度百分比: {(supercooling/heating_transition_temp)*100:.2f}%')
    
    plt.xlabel('Temperature (K)', fontsize=14, fontweight='bold')
    plt.ylabel(r'Volume ($\mathrm{\AA^3}$)', fontsize=14, fontweight='bold')
    plt.title('Temperature-Volume Curve: Phase Transition Analysis', fontsize=16, fontweight='bold')
    plt.legend(fontsize=9, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # 保存图形
    plt.savefig('temperature_volume_curve.png', dpi=300, bbox_inches='tight')
    print('\n图形已保存为 temperature_volume_curve.png')
    
    plt.show()

if __name__ == '__main__':
    main()
