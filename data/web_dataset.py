import tarfile
import csv
import os
from io import BytesIO
from collections import defaultdict

# CONFIG
OUTPUT_DIR = "webdataset_shards"
TSV_PATH = "dataset2.tsv"
IMAGES_PER_TAR = 100000            # Your dataset structure
SHARD_SIZE = 20000                 # Recommended

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------------
# Step 1: Load TSV and group requests by batch file
# ----------------------------------------------------------
print("Loading captions...", flush = True)

batch_requests = defaultdict(dict)   # {batch_idx: {filename: caption}}

with open(TSV_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t", fieldnames=["path", "caption"])

    for row in reader:
        fname = row["path"]             # e.g., "178234.jpg"
        caption = row["caption"]

        global_id = int(fname[:-4])
        batch_idx = (global_id // IMAGES_PER_TAR) + 1

        batch_requests[batch_idx][fname] = caption

print(f"Found {len(batch_requests)} batches with requested images.", flush = True)

# ----------------------------------------------------------
# Step 2: Helper to open output shards
# ----------------------------------------------------------
def open_new_shard(shard_idx):
    path = os.path.join(OUTPUT_DIR, f"shard-{shard_idx:06d}.tar")
    return tarfile.open(path, "w")

shard_idx = 0
sample_idx = 0
out_tar = open_new_shard(shard_idx)

# ----------------------------------------------------------
# Step 3: Process tar files
# ----------------------------------------------------------
tar_files = [f"images_batch_{i}.tar" for i in range(1, 21)]

for tar_name in tar_files:
    # extract batch index from name, assuming format images_batch_X.tar
    # adjust if your names differ
    batch_idx = int(tar_name.split("_")[-1].split(".")[0])

    if batch_idx not in batch_requests:
        continue   # No needed images in this tar

    needed = batch_requests[batch_idx]
    print(f"Processing {tar_name}, extracting {len(needed)} items...")

    tar_path = tar_name


    with tarfile.open(tar_path, "r") as in_tar:

        # Iterate only once through tar entries
        loop_index = 0
        for member in in_tar.getmembers():
            loop_index += 1
            if loop_index % 10000 == 0:
                print(f"  Processed {loop_index} entries...", flush=True)

            if not member.isfile():
                continue

            fn = os.path.basename(member.name)

            if fn not in needed:
                continue

            caption = needed[fn]
            img_bytes = in_tar.extractfile(member).read()

            # rotate output shard
            if sample_idx >= SHARD_SIZE:
                out_tar.close()
                shard_idx += 1
                sample_idx = 0
                out_tar = open_new_shard(shard_idx)

            # Write the image
            img_info = tarfile.TarInfo(name=f"{sample_idx:09d}.jpg")
            img_info.size = len(img_bytes)
            out_tar.addfile(img_info, BytesIO(img_bytes))

            # Write the caption
            cap_bytes = caption.encode("utf-8")
            cap_info = tarfile.TarInfo(name=f"{sample_idx:09d}.txt")
            cap_info.size = len(cap_bytes)
            out_tar.addfile(cap_info, BytesIO(cap_bytes))

            sample_idx += 1

# Final shard close
try:
    out_tar.close()
except:
    print("Already closed")
    pass
print("Done! WebDataset shards created.")
