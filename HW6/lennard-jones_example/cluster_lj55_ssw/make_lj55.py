# make_lj55.py
with open("lasp.str", "w") as f:
    f.write("PBC 30.000000 30.000000 30.000000 90.0 90.0 90.0\n")
    count = 1
    for x in range(4):
        for y in range(4):
            for z in range(4):
                if count > 55: 
                    break
                # 为了防止原子重叠，我们将原子按 2.5 的间距排成一个立体网格
                px = x * 2.5 + 10.0
                py = y * 2.5 + 10.0
                pz = z * 2.5 + 10.0
                # 按照 lasp.str 的固定格式写入 (使用 Au 符号占位，LJ势场只认拓扑)
                f.write(f"Au {px:12.6f} {py:12.6f} {pz:12.6f} CORE {count:4d} Au Au 0.000 {count:4d}\n")
                count += 1
print("成功生成 55 原子的 lasp.str！")
