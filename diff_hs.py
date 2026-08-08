import torch

REF_PT = "/data/ref_hs.pt"
FORK_PT = "/data/fork_hs.pt"
TARGET_IDS = (0, 128803, 671, 6102, 294, 8760, 344, 128804, 128822)
COS_THRESH = 0.99

ref = torch.load(REF_PT, map_location="cpu")
print("ref type:", type(ref).__name__, "len:", len(ref), "ref[0].shape:", tuple(ref[0].shape))

fk = torch.load(FORK_PT, map_location="cpu")
print("fork type:", type(fk).__name__, "num_keys:", len(fk))
print("fork keys (first 3):", list(fk.keys())[:3])

key = None
for k in fk:
    if tuple(k) == TARGET_IDS:
        key = k
        break
if key is None:
    # fallback: entry with full 43 layers
    for k, v in fk.items():
        if len(v) == 43:
            key = k
            break
if key is None:
    print("ERROR: no matching fork entry for target ids", TARGET_IDS)
    raise SystemExit(1)

fkv = fk[key]
print("matched fork key:", tuple(key), "layers:", len(fkv), "shape[0]:", tuple(fkv[0].shape))

n = min(len(ref), len(fkv))
cos_list, mse_list = [], []
for i in range(n):
    a = ref[i].reshape(-1).float()
    b = fkv[i].reshape(-1).float()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    mse = ((a - b) ** 2).mean().item()
    cos_list.append(cos)
    mse_list.append(mse)
    flag = "   <<< DIVERGENT" if cos < COS_THRESH else ""
    print(f"layer {i:2d}  cos={cos:.6f}  mse={mse:.6e}{flag}")

first_div = next((i for i, c in enumerate(cos_list) if c < COS_THRESH), None)
print("\nFIRST DIVERGENT LAYER:", first_div)
if first_div is not None and first_div > 0:
    print("  -> bug manifests in layer", first_div,
          "(input from layer", first_div - 1, "was still aligned)")
elif first_div == 0:
    print("  -> divergence at layer 0 => embedding / input projection / first attention")
