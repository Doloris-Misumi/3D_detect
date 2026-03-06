import os
import glob

base_dir = '/home/hongsheng/k_radar_dataset'
seqs = ['2', '3', '4']

for seq in seqs:
    sparse_cube_dir = os.path.join(base_dir, seq, 'sparse_cube')
    if not os.path.exists(sparse_cube_dir):
        print(f"Directory not found: {sparse_cube_dir}")
        continue
    
    files = glob.glob(os.path.join(sparse_cube_dir, 'polar3d_*.npy'))
    print(f"Found {len(files)} files in {seq} to rename")
    
    count = 0
    for f in files:
        dirname, basename = os.path.split(f)
        new_basename = basename.replace('polar3d_', 'cube_')
        new_path = os.path.join(dirname, new_basename)
        try:
            os.rename(f, new_path)
            count += 1
        except Exception as e:
            print(f"Error renaming {f}: {e}")
            
    print(f"Renamed {count} files in {seq}")

# Update train.txt
train_txt_path = '/home/hongsheng/RL_3DOD/resources/split/train.txt'
print(f"Updating {train_txt_path}...")

# Read existing to avoid duplicates
with open(train_txt_path, 'r') as f:
    existing_lines = set([l.strip() for l in f.readlines()])

new_lines = []
for seq in seqs:
    label_dir = os.path.join(base_dir, seq, 'info_label')
    if not os.path.exists(label_dir):
        print(f"Label directory not found: {label_dir}")
        continue
        
    label_files = sorted(glob.glob(os.path.join(label_dir, '*.txt')))
    print(f"Found {len(label_files)} label files in {seq}")
    
    for f in label_files:
        basename = os.path.basename(f) # e.g. 00031.txt
        # Format in train.txt is SequenceID,FileName
        # e.g. 2,00031.txt
        entry = f"{seq},{basename}"
        if entry not in existing_lines:
            new_lines.append(entry)

if new_lines:
    with open(train_txt_path, 'a') as f:
        for line in new_lines:
            f.write(f"{line}\n")
    print(f"Added {len(new_lines)} new entries to train.txt")
else:
    print("No new entries to add to train.txt")
