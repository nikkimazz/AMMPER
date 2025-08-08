import pandas as pd
import os

# === Constants ===
EV_TO_J = 1.60218e-19
MASS_KG = 2.62e-10  # 64 µm cube of water

# === File paths ===
ion_path = r"C:\Users\nmazzare\OneDrive - NASA\Documents\AMMPER-1\Ionevents_1.dat"
ele_path = r"C:\Users\nmazzare\OneDrive - NASA\Documents\AMMPER-1\Elevents_1.dat"

# === Load and sum function ===
def load_energy(file_path, label):
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return 0
    df = pd.read_csv(file_path, sep=r'\s+', comment='#', header=0)
    energy_eV = df['Energy'].sum()
    energy_J = energy_eV * EV_TO_J
    dose = energy_J / MASS_KG
    print(f"📄 {label}: {os.path.basename(file_path)}")
    print(f"⚡ Energy: {energy_eV:.2e} eV ({energy_J:.3e} J)")
    print(f"🧪 Dose: {dose:.3f} Gy\n")
    return energy_J

# === Run
print("🔬 Summing Dose from Ion and Electron Events:\n")
ion_energy_J = load_energy(ion_path, "Ion")
ele_energy_J = load_energy(ele_path, "Electron")

total_dose_Gy = (ion_energy_J + ele_energy_J) / MASS_KG
print(f"📊 ✅ Total Dose Combined: {total_dose_Gy:.3f} Gy")
