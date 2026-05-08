import csv
import os

def extract_qm9_global_properties(input_filepath: str, output_filepath: str) -> None:
    """
    Parses QM9 .xyz files to extract fixed-length global properties into a CSV.
    """
    headers = [
        "Dataset_ID", "A_GHz", "B_GHz", "C_GHz", "Dipole_Debye", 
        "Polarizability_Bohr3", "HOMO_Hartree", "LUMO_Hartree", "Gap_Hartree", 
        "R2_Bohr2", "ZPVE_Hartree", "U_298K_Hartree", "H_298K_Hartree", 
        "G_298K_Hartree", "Cv_cal_mol_K", "SMILES", "InChI"
    ]
    
    if not os.path.exists(input_filepath):
        raise FileNotFoundError(f"Target file missing: {input_filepath}")

    processed_count = 0
    with open(input_filepath, 'r', encoding='utf-8') as infile, \
         open(output_filepath, 'w', newline='', encoding='utf-8') as outfile:
        
        writer = csv.writer(outfile)
        writer.writerow(headers)
        
        while True:
            line1 = infile.readline()
            if not line1:
                break  # EOF
            
            line1 = line1.strip()
            if not line1:
                continue # Skip empty lines
                
            num_atoms = int(line1)
            
            # Parse Line 2: Properties
            prop_tokens = infile.readline().strip().split()
            if len(prop_tokens) < 16:
                raise ValueError(f"Malformed properties line at molecule {processed_count + 1}")
                
            dataset_id = f"{prop_tokens[0]}_{prop_tokens[1]}"
            scalar_props = prop_tokens[2:16]
            
            # Skip atom coordinate lines
            for _ in range(num_atoms):
                infile.readline()
                
            # Skip frequencies line
            infile.readline()
            
            # Parse SMILES and InChI
            smiles_tokens = infile.readline().strip().split()
            inchi_tokens = infile.readline().strip().split()
            
            smiles = smiles_tokens[0] if smiles_tokens else ""
            inchi = inchi_tokens[0] if inchi_tokens else ""
            
            row = [dataset_id] + scalar_props + [smiles, inchi]
            writer.writerow(row)
            processed_count += 1

    print(f"Extraction complete. Processed {processed_count} molecules.")

# Execution target:
input_path = r"E:\JupyterPjs\AI4Chem\HW2\QM9_2000.xyz"
output_path = r"E:\JupyterPjs\AI4Chem\HW2\QM9_global_properties.csv"
extract_qm9_global_properties(input_path, output_path)