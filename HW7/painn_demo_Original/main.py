import os, sys
import datetime
import random
import math
import torch
import torch.optim as optim
import numpy as np

from torch import amp
from model import PaiNN
from load_data import dataloader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_model(loss,best_loss,train_path,model):
    if loss < best_loss:
        best_loss = loss
        torch.save(model.state_dict(), train_path)
        print(f"[SAVE MODEL] {epoch} {best_loss}")
    return best_loss

def print_loss(_energy_out,_energy_y,_force_out,_force_y,node_num,start_time,end_time):
    energy_loss_mole = torch.sqrt(torch.mean((_energy_out - _energy_y) ** 2)).cpu().item()
    energy_loss_atom = torch.sqrt(torch.sum((_energy_out - _energy_y) ** 2 / node_num.view(-1, 1)) / torch.sum(node_num)).cpu().item()
    force_loss = torch.sqrt(torch.mean((_force_out - _force_y) ** 2)).cpu().item()
    print(f"[EPOCH] {epoch} [RMSE] [energy] [mol] {energy_loss_mole:.6f} [atom] {energy_loss_atom:.6f} [force] {force_loss:.6f} [time] {end_time - start_time}")
    energy_loss = torch.mean(torch.abs(_energy_out - _energy_y)).cpu().item()
    force_loss = torch.mean(torch.abs(_force_out - _force_y)).cpu().item()
    print(f"[EPOCH] {epoch} [MAE]  [energy] [mol] {energy_loss:.6f} [atom] -------- [force] {force_loss:.6f}")
    loss = force_loss + energy_loss_atom
    return loss


if __name__ == '__main__':
    set_seed(99)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("pid", os.getpid())
    batch_size = 32
    cutoff = 5.0
    train_valid_precent = [0.95,0.05]
    print("cutoff", cutoff)
    print("batch_size", batch_size)
    train_loader, valid_loader, info = dataloader(batch_size, train_valid_precent,cutoff)
    #######################################################################################################
    hidden_channel = 128
    num_layers = 3
    print("hidden_channel", hidden_channel)
    print("num_layers", num_layers)
    E_factor = 1
    F_factor = 20
    print("E_factor", E_factor)
    print("F_factor", F_factor)
    model = PaiNN(n_atom_basis=hidden_channel,n_interactions=num_layers)
    #######################################################################################################
    lr = 0.0001
    print("lr", lr)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    batch_count = math.ceil(info["train_count"] / batch_size)
    scheduler_lr = CosineAnnealingWarmRestarts(optimizer, T_0=batch_count, T_mult=8)
    #######################################################################################################
    model = model.to(device)
    print('参数总量:', sum(p.numel() for p in model.parameters()))
    criterion = torch.nn.SmoothL1Loss()
    train_path = f'./best_model.pth'

    max_clip = 2.0
    epoch_num = 500
    print("max_clip", max_clip)
    print("epoch_num", epoch_num)

    best_loss = float("inf")
    use_amp = device.type == "cuda"
    scaler = amp.GradScaler(enabled=use_amp)
    for epoch in range(epoch_num):
        model.train()
        start_time = datetime.datetime.now()
        _energy_out, _energy_y = torch.Tensor([]).to(device), torch.Tensor([]).to(device)
        _force_out, _force_y = torch.Tensor([]).to(device), torch.Tensor([]).to(device)
        node_num = torch.Tensor([]).to(device)
        for idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            # 混合精度训练
            with amp.autocast(device.type, enabled=use_amp):
                energy, force = model(data)
                energy_loss = criterion(energy, data.energy.view(-1, 1))
                force_loss = criterion(force, data.force)
                loss = energy_loss * E_factor + force_loss * F_factor

            scaler.scale(loss).backward()
            # clip
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_clip)
            # amp update
            scaler.step(optimizer)
            scaler.update()
            scheduler_lr.step()
            ####################################################################################
            _energy_out = torch.cat([_energy_out, energy.float().detach_().view(-1, 1)], dim=0)
            _energy_y = torch.cat([_energy_y, data.energy.view(-1, 1)], dim=0)
            _force_out = torch.cat([_force_out, force.float().detach_().view(-1, 3)], dim=0)
            _force_y = torch.cat([_force_y, data.force.view(-1, 3)], dim=0)

            node_num = torch.cat([node_num, data.i_num.view(-1, 1)], dim=0)

            if idx % 10 == 0:
                print(f"[IDX]:{idx:0>3} [loss]:{loss.item():.6f} [energy]:{energy_loss.item():.6f}  [force]:{force_loss:.6f} [lr]:{optimizer.param_groups[0]['lr']:.7f}")
        end_time = datetime.datetime.now()
        loss = print_loss(_energy_out,_energy_y,_force_out,_force_y,node_num,start_time,end_time)
        best_loss = save_model(loss,best_loss,train_path,model)

        model.eval()
        _energy_out, _energy_y = torch.Tensor([]).to(device), torch.Tensor([]).to(device)
        _force_out, _force_y = torch.Tensor([]).to(device), torch.Tensor([]).to(device)
        node_num = torch.Tensor([]).to(device)
        start_time = datetime.datetime.now()
        for idx, data in enumerate(valid_loader):
            data.to(device)
            with amp.autocast(device.type, enabled=use_amp):
                energy, force = model(data)
            ####################################################################################
            _energy_out = torch.cat([_energy_out, energy.float().detach_().view(-1, 1)], dim=0)
            _energy_y = torch.cat([_energy_y, data.energy.view(-1, 1)], dim=0)
            _force_out = torch.cat([_force_out, force.float().detach_().view(-1, 3)], dim=0)
            _force_y = torch.cat([_force_y, data.force.view(-1, 3)], dim=0)
            node_num = torch.cat([node_num, data.i_num.view(-1, 1)], dim=0)
        end_time = datetime.datetime.now()
        loss = print_loss(_energy_out,_energy_y,_force_out,_force_y,node_num,start_time,end_time)
