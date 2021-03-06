import sys  
import torch
sys.path.append("/tritonhomes/korbar/phd/detr_working_copy")
from datasets import calvin

failed = {"val": []}
ds_len = {}
for key in failed:
    ds = calvin(key)
    ds_len[key] = len(ds)
    for i in range(len(ds)):
        try:
            img, target = ds.__getitem__(i)
        except Exception as e: 
            print(str(e))
            print(i)
            failed[key].append(i)

print(ds_len)
print(len(failed['val']))
torch.save(failed, "CHAR_failed_val.pth")