import random

import torch
from torch_cluster import knn
from torch_geometric.nn import radius
from collections import defaultdict

PBC = torch.BoolTensor([True, True, True])
F_threshold = 1e-3

CACHE = {}

global HEALTH
HEALTH = defaultdict(int)


def gen_cartesian_prod(_x, _y, _z):
    _x = _x.item()
    _y = _y.item()
    _z = _z.item()

    if (_x, _y, _z) in CACHE: return CACHE[(_x, _y, _z)]
    x = torch.cat((torch.zeros(1), torch.arange(-1 * _x, 0), torch.arange(1, _x + 1)))
    y = torch.cat((torch.zeros(1), torch.arange(-1 * _y, 0), torch.arange(1, _y + 1)))
    z = torch.cat((torch.zeros(1), torch.arange(-1 * _z, 0), torch.arange(1, _z + 1)))
    cartesian_prod = torch.cartesian_prod(x, y, z)
    CACHE[(_x, _y, _z)] = cartesian_prod
    return cartesian_prod


def build_pbc(pos, cell, cutoff=5.0, pbc=PBC):
    ##############################################################
    inv_distances = torch.norm(cell.inverse().t(), dim=1)  # 轴的长度
    cell_xyz_repeats = torch.ceil(cutoff * inv_distances).long()
    cell_xyz_repeats = torch.where(pbc, cell_xyz_repeats, cell_xyz_repeats.new_zeros(()))  # 每个晶胞 x,y,z 扩胞次数
    cell_num = torch.prod(2 * cell_xyz_repeats + 1, dim=0)  # 每个晶胞扩胞总计次数
    ##############################################################
    _x, _y, _z = cell_xyz_repeats
    single_shift = torch.mm(gen_cartesian_prod(_x, _y, _z), cell.to(torch.float))
    shift = single_shift.repeat_interleave(pos.shape[0], 0)
    pos_j = pos.repeat((cell_num, 1)) + shift  # j的坐标
    ##############################################################
    i_num = torch.tensor(pos.shape[0])
    j_num = torch.tensor(shift.shape[0])  # j的个数
    return shift, cell_num, i_num, j_num, pos_j


def check_loop(egde, node):
    return egde.numel() == 1 and egde.item() == node


def check_healthy(data, force, MAX_DIST=5.0):
    edge_index = knn(data.pos_j, data.pos, 2)
    i, j = edge_index
    mask = i != j
    i = i[mask]
    j = j[mask]
    ##########################################################################距离超过cutoff
    dist = (data.pos_j[j] - data.pos[i]).norm(dim=-1)
    # 找到最大的距离
    max_dist = dist.max().item()
    # 按照最大的距离cutoff一次
    if max_dist > MAX_DIST:
        HEALTH["距离"] += 1
        print(f"\n[structure_id] {data.structure_id} [max-dist] {max_dist:.2f} 存在更大的cutoff半径")
        return False, None, HEALTH
    ##########################################################################完全对称求和为0
    edge_index = radius(data.pos_j, data.pos, r=max(MAX_DIST, max_dist))
    i, j = edge_index
    v_r = (data.pos_j[j] - data.pos[i]) * data.z[j % data.num_nodes].view(-1, 1)
    v_sum = torch.zeros((data.z.shape[0], 3))
    v_sum.index_add_(0, i, v_r)
    zero_v = torch.zeros((1, 3))
    exists = (v_sum == zero_v).all(dim=1).any().item()
    if exists:
        HEALTH["完全对称"] += 1
        print(f"\n[structure_id] {data.structure_id} {max_dist:.2f} 完全对称")
        return False, None, HEALTH
    ##########################################################################只跟自己连接
    for node in range(data.z.shape[0]):
        if check_loop(torch.unique(j[i == node] % data.num_nodes), node):
            HEALTH["自环"] += 1
            print(f"\n[structure_id] {data.structure_id} [node] {node} 自环")
            return False, None, HEALTH
    ##########################################################################不受力
    v_r = (data.pos_j[j] - data.pos[i]).abs()
    abs_v_sum = torch.zeros((data.z.shape[0], 3))
    abs_v_sum.index_add_(0, i, v_r)
    exists = (abs_v_sum == 0).any().item()
    if exists:
        ################################################################################不受力但是和F保持一致
        if torch.equal(abs_v_sum == 0, force == 0):
            HEALTH["非完全对称  不受力"] += 1
            print(f"\n[structure_id] {data.structure_id} 不受力")
            return True, None, HEALTH
        ###############################################################################不受力但是F很小
        force[(abs_v_sum == 0) & (force < F_threshold)] = 0
        if torch.equal(abs_v_sum == 0, force == 0):
            HEALTH["非完全对称 修改力"] += 1
            print(f"\n[structure_id] {data.structure_id} 修改受力")
            #return True, force, HEALTH
            return False, None, HEALTH
        HEALTH["非完全对称 异常"] += 1
        return False, None, HEALTH
    HEALTH["正常"] += 1
    return True, None, HEALTH


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
    import os, sys
    from load_data import PBCDataset

    FILE_PATH = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.abspath(os.path.join("FILE_PATH", "..", "..", "data", "train"))
    train_dataset = PBCDataset(train_path)
    check_healthy(train_dataset[25013])
