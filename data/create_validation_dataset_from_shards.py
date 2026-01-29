import tarfile
import os
from pathlib import Path
import csv

# Configuration
tar_files = [
    'webdataset_shards/shard-000060.tar',
    'webdataset_shards/shard-000061.tar',
    'webdataset_shards/shard-000062.tar',
    'webdataset_shards/shard-000063.tar',
    'webdataset_shards/shard-000064.tar',
    'webdataset_shards/shard-000065.tar'
]

output_dir = 'validation'
tsv_file = 'captions.tsv'

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Open TSV file for writing
with open(tsv_file, 'w', newline='', encoding='utf-8') as tsvfile:
    writer = csv.writer(tsvfile, delimiter='\t')
    writer.writerow(['path', 'caption'])
    
    global_index = 0
    
    # Process each tar file
    for tar_path in tar_files:
        print(f"Processing {tar_path}...")
        
        with tarfile.open(tar_path, 'r') as tar:
            members = tar.getmembers()
            
            # Build a dict of base names to their jpg and txt members
            pairs = {}
            for member in members:
                if member.isfile():
                    base_name = os.path.splitext(member.name)[0]
                    ext = os.path.splitext(member.name)[1]
                    
                    if base_name not in pairs:
                        pairs[base_name] = {}
                    
                    pairs[base_name][ext] = member
            
            # Process each pair in order
            for base_name in sorted(pairs.keys()):
                if '.jpg' in pairs[base_name] and '.txt' in pairs[base_name]:
                    jpg_member = pairs[base_name]['.jpg']
                    txt_member = pairs[base_name]['.txt']
                    
                    # Extract and save image
                    output_path = f'{global_index}.jpg'
                    
                    jpg_file = tar.extractfile(jpg_member)
                    with open(output_path, 'wb') as out_img:
                        out_img.write(jpg_file.read())
                    
                    # Extract caption
                    txt_file = tar.extractfile(txt_member)
                    caption = txt_file.read().decode('utf-8').strip()
                    
                    # Write to TSV
                    writer.writerow([output_path, caption])
                    
                    global_index += 1
                    
                    if global_index % 1000 == 0:
                        print(f"  Processed {global_index} images...")

print(f"\nDone! Extracted {global_index} images to '{output_dir}/' and created '{tsv_file}'")