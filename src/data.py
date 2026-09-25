"""
data.py - GameDiT data: Copernicus DEM GLO-30  ->  256x256 heightmap crops.

PROJECT-SPECIFIC. train.py only needs two things from this file:
    build_sampler(**kwargs) -> LabeledSampleable   (returns heightmaps + {"y": relief})
    visualize(x) -> ndarray b h w 3 in [0, 1]      (hillshade render for sample grids)

Run once, on a CPU machine:
    python data.py download     # ~180 tiles (~5 GB) into data/raw/
    python data.py preprocess   # resample, crop, filter, normalise -> data/processed/
    python data.py preview      # hillshade grid of real crops -> data/processed/preview.png
    python data.py all          # all three in order
    python data.py selftest     # offline check on synthetic terrain (no download)

Data: Copernicus DEM GLO-30 Public, https://registry.opendata.aws/copernicus-dem
(free under the Copernicus licence; see README for the required attribution).
"""
import argparse
import math
import os
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from paths import Condition, LabeledSampleable

BUCKET_URL = "https://copernicus-dem-30m.s3.amazonaws.com"
M_PER_DEG = 111_320.0  # metres per degree of latitude (approx.)

# Mountain regions: (lat_min, lat_max, lon_min, lon_max), inclusive.
# Values are the integer south-west corners of 1x1 degree tiles.
REGIONS: Dict[str, Tuple[int, int, int, int]] = {
    "alps":               (44, 47, 5, 13),
    "colorado_rockies":   (37, 39, -108, -105),
    "canadian_rockies":   (50, 52, -118, -115),
    "himalaya":           (27, 30, 81, 91),
    "karakoram":          (35, 36, 74, 77),
    "central_andes":      (-17, -14, -73, -70),
    "patagonia":          (-50, -47, -74, -72),
    "iceland":            (63, 65, -22, -15),
    "scottish_highlands": (56, 57, -6, -3),
    "nz_southern_alps":   (-45, -43, 168, 171),
}

# Relief (height range of a crop, metres) is mapped to y in [0, 1] on a log scale,
# because relief is heavily skewed (many gentle crops, few huge ones).
RELIEF_MIN_M = 150.0
RELIEF_MAX_M = 4000.0


# ---------------------------------------------------------------------------
# Tiles
# ---------------------------------------------------------------------------
def tile_name(lat: int, lon: int) -> str:
    """
    Name of the 1x1 degree tile whose south-west corner is (lat, lon).
    e.g. (45, 6) -> Copernicus_DSM_COG_10_N45_00_E006_00_DEM
    """
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"


def tile_url(name: str) -> str:
    return f"{BUCKET_URL}/{name}/{name}.tif"


def list_tiles() -> Dict[str, str]:
    """Returns {tile_name: region} for every tile in REGIONS."""
    tiles = {}
    for region, (lat0, lat1, lon0, lon1) in REGIONS.items():
        for lat in range(lat0, lat1 + 1):
            for lon in range(lon0, lon1 + 1):
                tiles[tile_name(lat, lon)] = region
    return tiles


def _download_one(name: str, raw_dir: str) -> str:
    path = os.path.join(raw_dir, name + ".tif")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return "cached"
    tmp = path + ".part"
    try:
        urllib.request.urlretrieve(tile_url(name), tmp)
        os.replace(tmp, path)
        return "ok"
    except urllib.error.HTTPError as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        # Ocean-only and unreleased tiles simply don't exist in the bucket
        if e.code in (403, 404):
            return "missing"
        return f"error {e.code}"
    except Exception as e:  # network hiccup: report and carry on
        if os.path.exists(tmp):
            os.remove(tmp)
        return f"error {type(e).__name__}"


def download(raw_dir: str = "data/raw", workers: int = 8):
    """Download every tile in REGIONS (skips files already on disk)."""
    os.makedirs(raw_dir, exist_ok=True)
    names = list(list_tiles().keys())
    print(f"Downloading {len(names)} tiles to {raw_dir} ...")
    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(lambda n: _download_one(n, raw_dir), names))
    counts: Dict[str, int] = {}
    for r in results:
        counts[r] = counts.get(r, 0) + 1
    print("Download summary:", counts)
    errors = [n for n, r in zip(names, results) if r.startswith("error")]
    if errors:
        print(f"{len(errors)} tiles failed - re-run `python data.py download` to retry them.")


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def read_tile(path: str) -> Tuple[np.ndarray, float]:
    """
    Returns:
        - dem: h w float32 (metres; NaN where no data)
        - lat_center: tile centre latitude (degrees)
    """
    import rasterio
    with rasterio.open(path) as src:
        dem = src.read(1).astype(np.float32)
        if src.nodata is not None:
            dem[dem == src.nodata] = np.nan
        lat_center = 0.5 * (src.bounds.top + src.bounds.bottom)
    return dem, lat_center


def resample_to_metres(dem: np.ndarray, lat_center: float, target_m: float) -> np.ndarray:
    """
    Resample a 1x1 degree tile so each pixel is ~target_m x target_m metres.
    (Longitude pixels shrink towards the poles; this makes pixels square again.)
    """
    H, W = dem.shape
    dy_m = M_PER_DEG / H
    dx_m = M_PER_DEG * math.cos(math.radians(lat_center)) / W
    new_h = max(1, round(H * dy_m / target_m))
    new_w = max(1, round(W * dx_m / target_m))
    t = torch.from_numpy(dem)[None, None]  # 1 1 h w
    out = F.interpolate(t, size=(new_h, new_w), mode="bilinear", antialias=True, align_corners=False)
    return out[0, 0].numpy()


def relief_to_y(relief_m: np.ndarray) -> np.ndarray:
    """Relief in metres -> condition y in [0, 1] (log scale)."""
    lo, hi = math.log(RELIEF_MIN_M), math.log(RELIEF_MAX_M)
    return np.clip((np.log(np.maximum(relief_m, 1e-3)) - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def y_to_relief(y: np.ndarray) -> np.ndarray:
    """Inverse of relief_to_y: condition y -> relief in metres (for export to UE5)."""
    lo, hi = math.log(RELIEF_MIN_M), math.log(RELIEF_MAX_M)
    return np.exp(lo + np.asarray(y) * (hi - lo))


def extract_crops(
    dem: np.ndarray,
    size: int = 256,
    stride: int = 128,
    min_relief: float = RELIEF_MIN_M,
    max_sea_frac: float = 0.05,
) -> Tuple[List[np.ndarray], List[float], Dict[str, int]]:
    """
    Slide a size x size window over the tile and keep the interesting crops.
    Each kept crop is normalised to [-1, 1]; its original height range is returned.
    Returns:
        - crops: list of h w float16 arrays in [-1, 1]
        - reliefs: list of heights ranges in metres
        - stats: how many crops were kept / rejected and why
    """
    H, W = dem.shape
    crops, reliefs = [], []
    stats = {"kept": 0, "nodata": 0, "sea": 0, "flat": 0}
    for i in range(0, H - size + 1, stride):
        for j in range(0, W - size + 1, stride):
            c = dem[i:i + size, j:j + size]
            if not np.isfinite(c).all():
                stats["nodata"] += 1
                continue
            if (c <= 0.0).mean() > max_sea_frac:
                stats["sea"] += 1
                continue
            lo, hi = float(c.min()), float(c.max())
            relief = hi - lo
            if relief < min_relief:
                stats["flat"] += 1
                continue
            crops.append(((c - lo) / relief * 2.0 - 1.0).astype(np.float16))
            reliefs.append(relief)
            stats["kept"] += 1
    return crops, reliefs, stats


def save_split(out_dir: str, split: str, crops: List[np.ndarray], reliefs: List[float],
               tiles: List[str], regions: List[str]):
    """Writes <split>.npy (n h w float16) and <split>_meta.npz."""
    os.makedirs(out_dir, exist_ok=True)
    relief_m = np.asarray(reliefs, dtype=np.float32)
    np.save(os.path.join(out_dir, f"{split}.npy"), np.stack(crops).astype(np.float16))
    np.savez(
        os.path.join(out_dir, f"{split}_meta.npz"),
        relief_m=relief_m,
        y=relief_to_y(relief_m),
        tile=np.asarray(tiles),
        region=np.asarray(regions),
    )


def preprocess(
    raw_dir: str = "data/raw",
    out_dir: str = "data/processed",
    target_m: float = 60.0,
    size: int = 256,
    stride: int = 128,
    val_frac: float = 0.05,
    seed: int = 0,
):
    """
    Tiles -> crops. The train/val split is done by TILE, not by crop:
    neighbouring crops overlap, so a random crop split would leak.
    """
    tile_regions = list_tiles()
    names = sorted(n for n in tile_regions if os.path.exists(os.path.join(raw_dir, n + ".tif")))
    assert names, f"No tiles found in {raw_dir} - run `python data.py download` first."

    rng = np.random.default_rng(seed)
    is_val = rng.random(len(names)) < val_frac
    if not is_val.any():
        is_val[rng.integers(len(names))] = True

    buckets = {s: {"crops": [], "reliefs": [], "tiles": [], "regions": []} for s in ("train", "val")}
    totals = {"kept": 0, "nodata": 0, "sea": 0, "flat": 0}
    per_region: Dict[str, int] = {}

    for k, (name, val) in enumerate(zip(names, is_val)):
        dem, lat_c = read_tile(os.path.join(raw_dir, name + ".tif"))
        dem = resample_to_metres(dem, lat_c, target_m)
        crops, reliefs, stats = extract_crops(dem, size=size, stride=stride)
        for key in totals:
            totals[key] += stats[key]
        region = tile_regions[name]
        per_region[region] = per_region.get(region, 0) + len(crops)

        b = buckets["val" if val else "train"]
        b["crops"] += crops
        b["reliefs"] += reliefs
        b["tiles"] += [name] * len(crops)
        b["regions"] += [region] * len(crops)
        print(f"[{k + 1}/{len(names)}] {name} ({region}): {stats['kept']} crops")

    for split, b in buckets.items():
        if b["crops"]:
            save_split(out_dir, split, b["crops"], b["reliefs"], b["tiles"], b["regions"])
            print(f"{split}: {len(b['crops'])} crops")

    print("Crop filter summary:", totals)
    print("Crops per region:", per_region)


# ---------------------------------------------------------------------------
# Sampler (what train.py uses)
# ---------------------------------------------------------------------------
def random_dihedral(z: torch.Tensor) -> torch.Tensor:
    """
    Random flips / 90-degree rotations, chosen independently per sample.
    (Terrain has no preferred orientation, so this is 8x free data.)
    Args:
        - z: b c h w
    Returns:
        - z: b c h w
    """
    b = z.shape[0]

    def coin():
        return torch.rand(b, 1, 1, 1, device=z.device) < 0.5

    z = torch.where(coin(), z.flip(-1), z)
    z = torch.where(coin(), z.flip(-2), z)
    if z.shape[-1] == z.shape[-2]:
        z = torch.where(coin(), z.transpose(-1, -2), z)
    return z


class HeightmapSampler(nn.Module, LabeledSampleable):
    """
    Holds the whole dataset as a buffer, so path.to("cuda") moves it onto the GPU
    (~2-3 GB in float16). Sampling a batch is then just GPU indexing - no dataloader.
    """
    def __init__(self, data_dir: str = "data/processed", split: str = "train", augment: bool = True):
        super().__init__()
        x = np.load(os.path.join(data_dir, f"{split}.npy"))            # n h w float16
        meta = np.load(os.path.join(data_dir, f"{split}_meta.npz"))
        self.register_buffer("data", torch.from_numpy(x), persistent=False)
        self.register_buffer("y", torch.from_numpy(meta["y"].astype(np.float32)), persistent=False)
        self.register_buffer("relief_m", torch.from_numpy(meta["relief_m"].astype(np.float32)), persistent=False)
        self.augment = augment

    def __len__(self) -> int:
        return self.data.shape[0]

    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Condition]:
        """
        Returns:
            - z: b 1 h w   (heightmaps in [-1, 1])
            - cond: {"y": b}   (relief in [0, 1])
        """
        idx = torch.randint(0, len(self), (num_samples,), device=self.data.device)
        z = self.data[idx].float().unsqueeze(1)
        if self.augment:
            z = random_dihedral(z)
        return z, {"y": self.y[idx]}


def build_sampler(data_dir: str = "data/processed", split: str = "train", augment: bool = True) -> HeightmapSampler:
    return HeightmapSampler(data_dir=data_dir, split=split, augment=augment)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def visualize(x: torch.Tensor, azimuth: float = 315.0, altitude: float = 45.0, z_factor: float = 12.0) -> np.ndarray:
    """
    Hillshade (light from the north-west) blended with a terrain colour map.
    Raw heightmaps look like grey fog; this makes ridges and valleys readable.
    Args:
        - x: b c h w in roughly [-1, 1] (first channel is used)
        - z_factor: vertical exaggeration of the shading
    Returns:
        - rgb: b h w 3 in [0, 1]
    """
    from matplotlib import colormaps

    h = x[:, 0].detach().float().cpu().numpy()
    d_row, d_col = np.gradient(h * z_factor, axis=(1, 2))
    slope = np.pi / 2.0 - np.arctan(np.hypot(d_row, d_col))
    aspect = np.arctan2(-d_row, d_col)
    az = np.deg2rad(360.0 - azimuth)
    alt = np.deg2rad(altitude)
    shade = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    shade = np.clip((shade + 1.0) / 2.0, 0.0, 1.0)

    colour = colormaps["terrain"](np.clip((h + 1.0) / 2.0 * 0.8 + 0.2, 0, 1))[..., :3]  # skip the sea-blue end
    return np.clip(colour * (0.35 + 0.65 * shade[..., None]), 0.0, 1.0)


def preview(data_dir: str = "data/processed", n: int = 16):
    from train import save_grid
    sampler = build_sampler(data_dir, "train", augment=False)
    z, cond = sampler.sample(n)
    path = os.path.join(data_dir, "preview.png")
    save_grid(z, path, visualize)
    r = sampler.relief_m
    print(f"{len(sampler)} train crops | relief min {r.min():.0f} m, median {r.median():.0f} m, max {r.max():.0f} m")
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Self-test (offline, synthetic terrain)
# ---------------------------------------------------------------------------
def _selftest():
    import tempfile
    import shutil

    torch.manual_seed(0)
    np.random.seed(0)

    # 1. Tile names
    assert tile_name(45, 6) == "Copernicus_DSM_COG_10_N45_00_E006_00_DEM"
    assert tile_name(-14, -73) == "Copernicus_DSM_COG_10_S14_00_W073_00_DEM"
    n_tiles = len(list_tiles())
    print(f"tile names OK ({n_tiles} tiles in REGIONS)")

    # 2. Fake 1x1 degree tile: smooth mountains, a strip of sea, a flat plateau
    coarse = torch.rand(1, 1, 24, 24)
    dem = F.interpolate(coarse, size=(1800, 1800), mode="bicubic", align_corners=False)[0, 0].numpy()
    dem = (dem * 2500 + 800).astype(np.float32)
    dem[:, :300] = 0.0            # sea on the west side
    dem[:600, 1200:] = 500.0      # flat plateau in the north-east

    dem_m = resample_to_metres(dem, lat_center=45.0, target_m=60.0)
    assert dem_m.shape[0] < dem_m.shape[1] * 1.6 and dem_m.shape[1] < 1800, "resample shape looks wrong"
    crops, reliefs, stats = extract_crops(dem_m)
    print(f"crop filter on synthetic tile: {stats}")
    assert stats["kept"] > 0 and stats["sea"] > 0 and stats["flat"] > 0
    c0 = crops[0].astype(np.float32)
    assert abs(c0.min() + 1) < 1e-2 and abs(c0.max() - 1) < 1e-2, "crops should span [-1, 1]"
    y = relief_to_y(np.asarray(reliefs))
    assert (0 <= y).all() and (y <= 1).all()
    assert np.allclose(y_to_relief(relief_to_y(np.array([500.0]))), 500.0, rtol=1e-4)
    print("resample / crop / filter / normalise OK")

    # 3. Sampler + augmentation
    tmp = tempfile.mkdtemp()
    try:
        save_split(tmp, "train", crops, reliefs, ["fake"] * len(crops), ["test"] * len(crops))
        sampler = build_sampler(tmp, "train")
        z, cond = sampler.sample(8)
        assert z.shape == (8, 1, 256, 256) and cond["y"].shape == (8,)
        assert z.min() >= -1.001 and z.max() <= 1.001
        # augmentation only rearranges pixels: sorted values are unchanged
        a = torch.from_numpy(crops[0].astype(np.float32))[None, None].repeat(16, 1, 1, 1)
        b = random_dihedral(a)
        assert torch.equal(a.flatten(1).sort(dim=1).values, b.flatten(1).sort(dim=1).values)
        print("sampler + dihedral augmentation OK")

        # 4. Hillshade
        rgb = visualize(z)
        assert rgb.shape == (8, 256, 256, 3) and rgb.min() >= 0 and rgb.max() <= 1
        try:
            from train import save_grid
            save_grid(z, os.path.join(tmp, "preview.png"), visualize)
            assert os.path.exists(os.path.join(tmp, "preview.png"))
        except ImportError:
            pass
        print("hillshade visualisation OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("data.py OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["download", "preprocess", "preview", "all", "selftest"])
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--out_dir", default="data/processed")
    parser.add_argument("--target_m", type=float, default=60.0, help="metres per pixel after resampling")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if args.command == "selftest":
        _selftest()
    if args.command in ("download", "all"):
        download(args.raw_dir, args.workers)
    if args.command in ("preprocess", "all"):
        preprocess(args.raw_dir, args.out_dir, args.target_m, args.size, args.stride)
    if args.command in ("preview", "all"):
        preview(args.out_dir)