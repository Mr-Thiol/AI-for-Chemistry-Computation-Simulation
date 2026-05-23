import numpy as np
import math
from ase.io import read

def lat2abc(lat):
    a = np.linalg.norm(lat[0])
    b = np.linalg.norm(lat[1])
    c = np.linalg.norm(lat[2])
    alpha = math.acos(np.dot(lat[1], lat[2]) / (b * c)) * 180.0 / np.pi
    beta = math.acos(np.dot(lat[0], lat[2]) / (a * c)) * 180.0 / np.pi
    gamma = math.acos(np.dot(lat[0], lat[1]) / (a * b)) * 180.0 / np.pi
    return [a, b, c, alpha, beta, gamma]

def readtraj(input_traj_file, output_arc_file):
    """
    Convert trajectory file to ARC format file.
    
    Args:
        input_traj_file (str): Path to input trajectory file (e.g., './log/opt.traj')
        output_arc_file (str): Path to output ARC file (e.g., './log/opt.arc')
    """
    traj = read(input_traj_file, index=':')
    
    with open(output_arc_file, 'w') as fout:
        fout.write("!BIOSYM archive 2\nPBC=ON\n")
        
        for istr, atoms in enumerate(traj):
            elements = atoms.get_chemical_symbols()
            positions = atoms.get_positions()
            cell = atoms.get_cell()
            a, b, c, alpha, beta, gamma = lat2abc(cell)
            energy = atoms.get_potential_energy()
            
            fout.write("\t\t\t\tEnergy\t%8d        0.0000 %17.6f\n" % (istr, energy))
            fout.write("!DATE\n")
            fout.write("PBC%15.9f%15.9f%15.9f%15.9f%15.9f%15.9f\n" % (
                a, b, c, alpha, beta, gamma))
            
            for i, ele in enumerate(elements):
                fout.write("%-2s%18.9f%15.9f%15.9f CORE %4d %-2s %-2s %8.4f %4d\n" % (
                    ele, positions[i][0], positions[i][1], positions[i][2],
                    i + 1, ele, ele, 0.0, i + 1))
            
            fout.write("end\nend\n")

# Example usage:
# readtraj('./log/opt.traj', './log/opt.arc')
