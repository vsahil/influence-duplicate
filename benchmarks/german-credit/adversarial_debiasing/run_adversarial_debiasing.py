import os
for split in range(20):     # 20 data splits
    os.system(f"python adversarial_debiasing.py {split}")
print("Done!")