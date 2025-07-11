import os

print("Step 1: Generate synthetic data")
os.system("python data/generate_data.py")

print("Step 2: Run GMT baseline")
os.system("python models/train_gmt.py")

print("Step 3: Run MaCoDE baseline")
os.system("python models/train_macode.py")

print("Step 4: Run FoNE baseline")
os.system("python models/train_fone.py")
