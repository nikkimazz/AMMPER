import pandas as pd
import os

# === File paths ===
base_path = r"C:\Users\nmazzare\OneDrive - NASA\Documents\AMMPER-1"
old_path = os.path.join(base_path, "radiationData", "1000", "Track0")

# Proton 1 GeV OLD files
old_ele_path = os.path.join(old_path, "Elevents.dat")
old_ion_path = os.path.join(old_path, "Ionevents.dat")

# NEW files you uploaded (in AMMPER-1 root directory)
new_ele_path = os.path.join(base_path, "Elevents_0.dat")
new_ion_path = os.path.join(base_path, "Ionevents_0.dat")

# === Helper function ===
def load_energy_column(filepath):
    try:
        df = pd.read_csv(filepath, sep=r'\s+', header=None, comment='#')
        energy_raw = df.iloc[:, 3]  # 4th column = energy deposited (keV)
        energy = pd.to_numeric(energy_raw, errors='coerce').dropna()
        return energy
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return pd.Series(dtype=float)

# === Load data ===
old_ele_energy = load_energy_column(old_ele_path)
old_ion_energy = load_energy_column(old_ion_path)
new_ele_energy = load_energy_column(new_ele_path)
new_ion_energy = load_energy_column(new_ion_path)

# === Print total energy comparison ===
print("=== Total Energy Deposition Comparison (keV) ===")
print(f"Old Elevents.dat:   {old_ele_energy.sum():.2f} keV from {len(old_ele_energy)} events")
print(f"New Elevents_0.dat: {new_ele_energy.sum():.2f} keV from {len(new_ele_energy)} events\n")

print(f"Old Ionevents.dat:   {old_ion_energy.sum():.2f} keV from {len(old_ion_energy)} events")
print(f"New Ionevents_0.dat: {new_ion_energy.sum():.2f} keV from {len(new_ion_energy)} events")

# === Optional: Histogram comparison (commented) ===
# import matplotlib.pyplot as plt
# plt.hist(old_ele_energy, bins=50, alpha=0.5, label="Old Ele")
# plt.hist(new_ele_energy, bins=50, alpha=0.5, label="New Ele")
# plt.legend()
# plt.title("Electron Energy Deposition Comparison")
# plt.xlabel("Energy (keV)")
# plt.ylabel("Event Count")
# plt.show()
