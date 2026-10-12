import pandas as pd
import geopandas as gpd
from pathlib import Path
from datetime import datetime
import os
from filelock import FileLock

SCHEMA = {
    'region_id': str,
    'name': str,
    'country_code': str,
    'continent': str,
    'gadm_file': str,
    'gadm_name': str,
    'nuts_id': str,
    'resolution': int,
    'max_depth': int,
    'threshold': float,
    'year': int,
    'status': str,
    'output_path': str,
    'n_points': 'Int64',
    'classes': object,
    'created_at': str,
    'updated_at': str,
    'slurm_job_id': str,
    'error_msg': str,
    'gadm_level': int,
    'sampling': str
}

def name_col(level: int) -> str:
    """GADM stores the level-0 name in 'COUNTRY'; levels 1+ use 'NAME_{level}'."""
    return 'COUNTRY' if level == 0 else f'NAME_{level}'

class RegionConfig:
    """Single region configuration."""
    
    def __init__(self, row: pd.Series):
        for col in SCHEMA:
            setattr(self, col, row.get(col))

    def get_polygon(self):
        level = 2 if pd.isna(self.gadm_level) else int(self.gadm_level)
        gdf = gpd.read_file(self.gadm_file, layer=level)
        gid_col = f'GID_{level}'

        if level == 0:
            gid = self.region_id
        else:
            parts = self.region_id.split('_')
            gid = parts[0] + '.' + '.'.join(parts[1:-1]) + '_' + parts[-1]

        region = gdf[gdf[gid_col] == gid]
        if len(region) == 0:
            raise ValueError(f"{self.region_id}: GID {gid} not found in layer {level}")
        return region.union_all()
    
    def checkpoint_dir(self, storage_root: str) -> str:
        return f"{storage_root}/checkpoints/{self.region_id}"
    
    def __repr__(self):
        return f"RegionConfig({self.region_id} — {self.name}, status={self.status})"


class RegionIndex:
    """
    Manages the region index parquet file.
    
    Usage:
        idx = RegionIndex()
        idx.add(region_id='PL21', name='Małopolskie', ...)
        idx.pending()   # all pending regions
        idx.complete()  # all complete regions
    """
    
    def __init__(self, path: str = None):
        self.path = path or os.getenv('LUCERA_REGION_INDEX', 'data/index.parquet')
        self._df = self._load()
    
    def _load(self) -> pd.DataFrame:
        if Path(self.path).exists():
            return pd.read_parquet(self.path)

        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({col: pd.Series(dtype=dtype) 
                          for col, dtype in SCHEMA.items()})
        df.to_parquet(self.path, index=False)
        return df
    
    def _save(self):
        tmp = f"{self.path}.tmp.{os.getpid()}"
        self._df.to_parquet(tmp, index=False)
        os.replace(tmp, self.path) 
    
    def add(
        self,
        region_id: str,
        name: str,
        country_code: str,
        continent: str,
        gadm_file: str,
        gadm_name: str,
        gadm_level: int = 1,
        resolution: int = 7,
        max_depth: int = 2,
        threshold: float = 0.8,
        year: int = 2023,
        nuts_id: str = None,
        output_path: str = None,
        overwrite: bool = False,
        sampling: str = 'centroid'
    ) -> 'RegionIndex':
        if region_id in self._df['region_id'].values:
            if not overwrite:
                print(f"{region_id} already exists — skipping. Use overwrite=True to replace.")
                return self
            self._df = self._df[self._df['region_id'] != region_id]
        
        storage_root = os.getenv('LUCERA_STORAGE', 'data/')
        if output_path is None:
            output_path = f"{storage_root}/raw/{continent}/{country_code.lower()}/{region_id}/adaptive_r{resolution}_{year}.parquet"
        
        now = datetime.utcnow().isoformat()
        row = {
            'region_id': region_id,
            'name': name,
            'country_code': country_code,
            'continent': continent,
            'gadm_file': gadm_file,
            'gadm_name': gadm_name,
            'nuts_id': nuts_id,
            'resolution': resolution,
            'max_depth': max_depth,
            'threshold': threshold,
            'year': year,
            'status': 'pending',
            'output_path': output_path,
            'n_points': None,
            'classes': None,
            'created_at': now,
            'updated_at': now,
            'slurm_job_id': None,
            'error_msg': None,
            'sampling': sampling,
            'gadm_level': gadm_level,
        }
        self._df = pd.concat([self._df, pd.DataFrame([row])], ignore_index=True)
        self._save()
        print(f"Added {region_id} — {name}")
        return self
    
    def update(self, region_id: str, **kwargs) -> 'RegionIndex':
        with FileLock(f"{self.path}.lock", timeout=120):
            self._df = pd.read_parquet(self.path)
            mask = self._df['region_id'] == region_id
            if not mask.any():
                raise ValueError(f"Region {region_id} not found")
            kwargs['updated_at'] = datetime.utcnow().isoformat()
            for key, val in kwargs.items():
                if isinstance(val, list):
                    val = str(val)
                self._df.loc[mask, key] = val
            self._save()
        return self
    
    def get(self, region_id: str) -> RegionConfig:
        row = self._df[self._df['region_id'] == region_id]
        if len(row) == 0:
            raise ValueError(f"Region {region_id} not found")
        return RegionConfig(row.iloc[0])
    
    def pending(self) -> pd.DataFrame:
        return self._df[self._df['status'] == 'pending']
    
    def complete(self) -> pd.DataFrame:
        return self._df[self._df['status'] == 'complete']
    
    def failed(self) -> pd.DataFrame:
        return self._df[self._df['status'] == 'failed']
    
    def running(self) -> pd.DataFrame:
        return self._df[self._df['status'] == 'running']
    
    def summary(self):
        print(f"Total regions: {len(self._df)}")
        print(self._df['status'].value_counts().to_string())
        print(f"\nCountries: {self._df['country_code'].unique().tolist()}")
    
    def missing_class(self, label: int) -> pd.DataFrame:
        """Find complete regions missing a specific WorldCover class."""
        complete = self.complete()
        missing = complete[complete['classes'].apply(
            lambda c: c is None or label not in (c if isinstance(c, list) else [])
        )]
        return missing
    
    def __len__(self):
        return len(self._df)
    
    def __repr__(self):
        return f"RegionIndex({len(self._df)} regions, path={self.path})"