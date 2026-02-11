import os
import gzip
import shutil

# Source: AI/Bosma22a folder
source_dir = r"F:\MOD002691 - FP\pi_cai_project\picai_labels\csPCa_lesion_delineations\AI\Bosma22a"

# Destination: New subfolder
dest_dir = os.path.join(source_dir, "decompressed_files")

# Create destination folder
os.makedirs(dest_dir, exist_ok=True)

print(f"Scanning: {source_dir}")

# Only grab files that actually need decompressing (.gz)
# If a file is already .nii, we can just copy it (optional) or ignore it
files_to_process = [f for f in os.listdir(source_dir) if f.endswith(".gz") or f.endswith(".nii") or f.endswith(".nii.gz")]

print(f"Found {len(files_to_process)} compressed files (.gz).")
print(f"Saving to: {dest_dir}")

for i, filename in enumerate(files_to_process):
    src_path = os.path.join(source_dir, filename)
    
    # Define output filename (remove last 3 chars '.gz')
    # e.g. "10000.nii.gz" -> "10000.nii"
    new_filename = filename[:-3] 
    dest_path = os.path.join(dest_dir, new_filename)
    
    print(f"[{i+1}/{len(files_to_process)}] Decompressing {filename} -> {new_filename}...", end="")
    
    try:
        with gzip.open(src_path, 'rb') as f_in:
            with open(dest_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print("Done")
        
    except Exception as e:
        print(f" Error: {e}")

print(f"\nDecompression complete. Files are in:\n{dest_dir}")
