import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
from utilities.region_manager import RegionIndex
from scripts.build_region import build_region

def build_all(country_code: str = None, continent: str = None):
    idx = RegionIndex()
    pending = idx.pending()
    
    if country_code:
        pending = pending[pending['country_code'] == country_code]
    if continent:
        pending = pending[pending['continent'] == continent]
    
    print(f"Building {len(pending)} regions...")
    
    for _, region in pending.iterrows():
        try:
            build_region(region['region_id'])
        except Exception as e:
            print(f"Failed {region['region_id']}: {e}")
            continue

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--country-code')
    parser.add_argument('--continent')
    args = parser.parse_args()
    build_all(args.country_code, args.continent)