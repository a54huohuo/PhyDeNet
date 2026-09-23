"""vtu_to_combined.py — dataset preprocessing: Zenodo VTU pairs -> *_combined.csv

Converts the public Carotid Bifurcation Haemodynamics Dataset (Eulzer et al.,
Zenodo DOI 10.5281/zenodo.10695923) into the per-case point-cloud CSVs that
PhyDeNet trains on. For every matching pair

    {case}_wss.vtu    wall-surface mesh carrying the WSS field
    {case}_fluid.vtu  volumetric mesh carrying velocity / pressure fields

it writes {case}_combined.csv with columns  x, y, z, velocity, pressure, wss
[mm, m/s, Pa, Pa]:

  * coordinates come from the fluid mesh nodes,
  * velocity = norm of the 3-component velocity vector field (systolic),
  * pressure = the scalar pressure field,
  * wss = |WSS vector| transferred from the wall mesh to fluid nodes whose
    coordinates coincide within 1e-6 mm (cKDTree nearest-neighbour match);
    unmatched (interior) nodes keep wss = 0, which is also how the training
    pipeline identifies wall points (no-slip, u < 1e-6 m/s).

Usage:
  python preprocess/vtu_to_combined.py --src-dir /path/to/carotid_flow_database \
                                       --out-dir data/

Dependencies: pyvista, numpy, pandas, scipy
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import pyvista as pv
from scipy.spatial import cKDTree

MATCH_TOLERANCE_MM = 1e-6


def find_file_pairs(src_dir):
    """Find all ({case}_wss.vtu, {case}_fluid.vtu) pairs in src_dir."""
    pairs = []
    for wss_file in sorted(glob.glob(os.path.join(src_dir, '*_wss.vtu'))):
        fluid_file = wss_file.replace('_wss.vtu', '_fluid.vtu')
        if os.path.exists(fluid_file):
            pairs.append((wss_file, fluid_file))
        else:
            print(f'[warn] no matching fluid file for {wss_file}')
    return pairs


def process_file_pair(wss_file, fluid_file, output_dir):
    """Convert one VTU pair into {case}_combined.csv; returns the output path."""
    case = os.path.basename(wss_file).replace('_wss.vtu', '')
    print(f'== {case} ==')
    wss_mesh = pv.read(wss_file)
    fluid_mesh = pv.read(fluid_file)

    wss_points = np.asarray(wss_mesh.points)
    fluid_points = np.asarray(fluid_mesh.points)

    # --- transfer WSS from the wall mesh onto coincident fluid nodes ---
    tree = cKDTree(wss_points)
    distances, indices = tree.query(fluid_points, k=1)
    match = distances < MATCH_TOLERANCE_MM
    n_match = int(match.sum())

    wss_values = np.zeros(len(fluid_points))
    if n_match:
        wss_field = None
        for name in wss_mesh.point_data:
            if 'wss' in name.lower():
                wss_field = wss_mesh.point_data[name]
                break
        if wss_field is None:
            raise RuntimeError(f'{case}: no WSS field found on the wall mesh')
        if wss_field.ndim == 2 and wss_field.shape[1] == 3:
            wss_field = np.linalg.norm(wss_field, axis=1)
        wss_values[match] = wss_field[indices[match]]
    else:
        raise RuntimeError(f'{case}: no coordinate matches within tolerance')

    # --- velocity magnitude and pressure from the fluid mesh ---
    velocity = None
    pressure = None
    for name in fluid_mesh.point_data:
        data = fluid_mesh.point_data[name]
        if velocity is None and data.ndim == 2 and data.shape[1] == 3 \
                and ('velocity' in name.lower() or 'vel' in name.lower()):
            velocity, velocity_name = data, name
        if pressure is None and data.ndim == 1 \
                and ('pressure' in name.lower() or 'pres' in name.lower()):
            pressure, pressure_name = data, name
    if velocity is None or pressure is None:
        raise RuntimeError(f'{case}: velocity/pressure field not found')

    df = pd.DataFrame({
        'x': fluid_points[:, 0],
        'y': fluid_points[:, 1],
        'z': fluid_points[:, 2],
        'velocity': np.linalg.norm(velocity, axis=1),
        'pressure': pressure,
        'wss': wss_values,
    })

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'{case}_combined.csv')
    df.to_csv(out_path, index=False)
    n_wall = int((df.wss != 0).sum())
    print(f'   points={len(df):,}  wall={n_wall:,} ({n_wall/len(df):.1%})  '
          f'fields: {velocity_name}, {pressure_name}  -> {out_path}')
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--src-dir', required=True,
                        help='directory with {case}_wss.vtu / {case}_fluid.vtu pairs')
    parser.add_argument('--out-dir', default='data',
                        help='output directory for {case}_combined.csv (default: data/)')
    args = parser.parse_args()

    pairs = find_file_pairs(args.src_dir)
    if not pairs:
        raise SystemExit(f'no VTU pairs found in {args.src_dir}')
    print(f'{len(pairs)} file pairs found')

    ok, fail = 0, 0
    for wss_file, fluid_file in pairs:
        try:
            process_file_pair(wss_file, fluid_file, args.out_dir)
            ok += 1
        except Exception as e:
            print(f'[error] {os.path.basename(wss_file)}: {e}')
            fail += 1
    print(f'\ndone: {ok} converted, {fail} failed')
    raise SystemExit(1 if fail else 0)


if __name__ == '__main__':
    main()
