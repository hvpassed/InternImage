import os
import shutil
import random

# Define paths
source_dir = 'data/train/imgs'
train_dir = 'data/train'
val_dir = 'data/val'



# Get list of all files in source directory
all_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
# Shuffle the files
random.shuffle(all_files)

# Split the files into 80% train and 20% validation
split_index = int(0.8 * len(all_files))
train_files = all_files[:split_index]
val_files = all_files[split_index:]
print(all_files)

for file in val_files:
    newname = file.replace('.jpg', '_m.png')
    shutil.move(os.path.join(source_dir, file), os.path.join(val_dir+"/imgs", file))
    shutil.move(os.path.join(train_dir+"/masks", newname), os.path.join(val_dir+"/masks", newname))
# print(f"Moved {len(train_files)} files to {train_dir}")
# print(f"Moved {len(val_files)} files to {val_dir}")