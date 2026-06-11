import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import ee
import pandas as pd
from utilities.region_manager import RegionIndex
from utilities.grid import H3GridBuilder
from utilities.data_sources import AlphaEarth, ESAWorldCover
from utilities.data_builder import DatasetBuilder

def build_region(region_id: str, dry_run: bool = False):
    idx = RegionIndex()
    region = idx.get(region_id)
    
    if region.status == 'complete':
        print(f"{region_id} already complete — skipping")
        return
    
    if Path(region.output_path).exists():
        print(f"Output exists for {region_id} — marking complete")
        df = pd.read_parquet(region.output_path)
        idx.update(region_id,
            status='complete',
            n_points=len(df),
            classes=str(sorted(df['label'].dropna().unique().tolist()))
        )
        return
    
    print(f"Building {region_id} — {region.name}")
    idx.update(region_id, status='running')
    
    try:
        ee.Initialize(project='lucera-494618')
        polygon = region.get_polygon()
        
        if dry_run:
            grid = H3GridBuilder(polygon=polygon, resolution=region.resolution)
            print(f"Dry run — {len(grid)} cells, output: {region.output_path}")
            idx.update(region_id, status='pending')
            return
        
        checkpoint_dir = f"data/checkpoints/{region_id}"
        
        dataset = (
            DatasetBuilder()
            .from_grid(H3GridBuilder(polygon=polygon, resolution=region.resolution))
            .set_label_source(ESAWorldCover())
            .add_source(AlphaEarth(year=region.year))
            .build(
                strategy='adaptive',
                threshold=region.threshold,
                max_depth=region.max_depth,
                checkpoint_dir=checkpoint_dir,
                sampling=region.sampling
            )
        )
        
        Path(region.output_path).parent.mkdir(parents=True, exist_ok=True)
        dataset.save(region.output_path, format='parquet')
        
        df = dataset.data
        idx.update(region_id,
            status='complete',
            n_points=len(df),
            classes=str(sorted(df['label'].dropna().unique().tolist()))
        )
        print(f"Done — {region_id}: {len(df)} points")
        
    except Exception as e:
        idx.update(region_id, status='failed', error_msg=str(e))
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--region-id', required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    build_region(args.region_id, dry_run=args.dry_run)