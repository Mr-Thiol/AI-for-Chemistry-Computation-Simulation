import os, sys
import torch

from torch_geometric.nn import radius


def radius_graph(data, cutoff):
    ##############################################################################################batch_j
    atom_repeat = torch.index_select(data.cell_num, dim=0, index=data.batch)  # 每个原子扩胞多少次
    batch_j = data.batch.repeat_interleave(atom_repeat).contiguous()  # 扩胞后的原子归属batch
    ##############################################################################################radius 雷达图cutoff
    edge_index = radius(data.pos_j, data.pos, r=cutoff, batch_x=batch_j, batch_y=data.batch, max_num_neighbors=512)
    ###########################
    j, i = edge_index[1], edge_index[0]
    #############################################################################################mask
    i_num, j_num = data.i_num, data.j_num
    i_consum = torch.cumsum(torch.cat([torch.zeros(1, device=i_num.device), i_num[:-1]], dim=0), dim=0)
    _i_consum = i_consum.repeat_interleave(j_num)  # 用于计算_j->j后 + 上一个原子数
    i_consum = i_consum.repeat_interleave(i_num)
    mask_i = i - i_consum[i]
    j_consum = torch.cumsum(torch.cat([torch.zeros(1, device=j_num.device), j_num[:-1]], dim=0), dim=0)
    j_consum = j_consum.repeat_interleave(j_num)  # 对j的累加 [j-j_consum]%j_num
    j_num = j_num.repeat_interleave(j_num)
    mask_j = torch.remainder(j - j_consum[j], j_num[j])
    mask = mask_i != mask_j
    #############################################################################################real j i
    # j i 是对应的每个node的索引   _j 是pos_j的索引
    _j, i = j[mask], i[mask]
    ####################### 将j映射到i 以便 x[j]
    j = (torch.remainder(_j - j_consum[_j], (i_num.repeat_interleave(data.j_num)[_j])) + _i_consum[_j]).long()
    edge_index = torch.stack([j, i], dim=0)
    return edge_index, data.shift[_j], _j


if __name__ == '__main__':
    pass
