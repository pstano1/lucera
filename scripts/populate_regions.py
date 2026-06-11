import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import geopandas as gpd
from utilities.region_manager import RegionIndex

def populate(
    gadm_file: str,
    country_code: str,
    continent: str,
    gadm_level: int = 2,
    resolution: int = 7,
    max_depth: int = 2,
    threshold: float = 0.8,
    year: int = 2023,
    sampling: str = 'children',
    dry_run: bool = False
):
    gdf = gpd.read_file(gadm_file, layer=gadm_level)
    print(f"Found {len(gdf)} regions at level {gadm_level}")
    print(gdf[['NAME_1', 'NAME_2', 'HASC_2']].head(5))
    
    if dry_run:
        print("Dry run — not adding to index")
        return
    
    idx = RegionIndex()
    
    for _, row in gdf.iterrows():
        parent = row['NAME_1'].lower().replace(' ', '_')
        name = row[f'NAME_{gadm_level}']
        hasc = row[f'GID_{gadm_level}'].replace('.', '_')
        
        idx.add(
            region_id=hasc,
            name=name,
            country_code=country_code,
            continent=continent,
            gadm_file=gadm_file,
            gadm_name=name,
            gadm_level=gadm_level,
            resolution=resolution,
            max_depth=max_depth,
            threshold=threshold,
            year=year,
            sampling=sampling,
            output_path=f"data/raw/{continent}/{country_code.lower()}/{parent}/{hasc}_adaptive_r{resolution}_{year}.parquet"
        )
    
    idx.summary()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gadm-file', required=True)
    parser.add_argument('--country-code', required=True)
    parser.add_argument('--continent', required=True)
    parser.add_argument('--gadm-level', type=int, default=2)
    parser.add_argument('--resolution', type=int, default=7)
    parser.add_argument('--max-depth', type=int, default=2)
    parser.add_argument('--threshold', type=float, default=0.8)
    parser.add_argument('--year', type=int, default=2023)
    parser.add_argument('--sampling', default='children')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    
    populate(
        gadm_file=args.gadm_file,
        country_code=args.country_code,
        continent=args.continent,
        gadm_level=args.gadm_level,
        resolution=args.resolution,
        max_depth=args.max_depth,
        threshold=args.threshold,
        year=args.year,
        sampling=args.sampling,
        dry_run=args.dry_run
    )