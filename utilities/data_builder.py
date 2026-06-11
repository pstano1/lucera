from .data_sources import EarthEngineSource
from .grid import GridBuilder
import pandas as pd
import h3
from pathlib import Path
from tqdm import tqdm

class DatasetBuilder:
    """
    Orchestrates grid + sources into a dataset.
    
    Usage:
        dataset = (
            DatasetBuilder()
            .from_grid(H3GridBuilder(bbox=bbox_pl2, resolution=7))
            .add_source(AlphaEarth(year=2023))
            .add_source(ESAWorldCover())
            .build()
        )
    """
    
    def __init__(self):
        self._grid = None
        self._label_source = None
        self._sources = []
        self._df = None
    
    def from_grid(self, *grids: 'GridBuilder') -> 'DatasetBuilder':
        if len(grids) == 1:
            self._grid = grids[0]
        else:
            from utilities.grid import H3GridBuilder
            combined = H3GridBuilder.__new__(H3GridBuilder)
            combined.resolution = grids[0].resolution
            combined._cells = set()
            for g in grids:
                if hasattr(g, '_cells'):
                    combined._cells.update(g._cells)
            self._grid = combined
        return self
    
    def set_label_source(self, source: 'EarthEngineSource') -> 'DatasetBuilder':
        self._label_source = source
        return self
    
    def add_source(self, source: 'EarthEngineSource') -> 'DatasetBuilder':
        self._sources.append(source)
        return self
    
    def build(
        self,
        strategy: str = 'standard',
        threshold: float = 0.8,
        max_depth: int = 2,
        checkpoint_dir: str = 'data/checkpoints',
        sampling: str = 'centroid'
    ) -> 'DatasetBuilder':
        """
        Build dataset.
        
        Args:
            strategy: 'standard' | 'homogeneous' | 'adaptive'
            threshold: homogeneity threshold (0-1)
            max_depth: max subdivision depth for adaptive strategy
        """
        import pandas as pd
        from utilities.grid import H3GridBuilder, KroneckerGridBuilder
        
        if self._grid is None:
            raise ValueError("No grid set — call from_grid() first")
        if not self._sources and self._label_source is None:
            raise ValueError("No sources set")
        
        if strategy == 'standard':
            self._df = self._build_standard()
        elif strategy == 'homogeneous':
            self._df = self._build_homogeneous(threshold)
        elif strategy == 'adaptive':
            self._df = self._build_adaptive(threshold, max_depth, checkpoint_dir, sampling)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        print(f"Dataset built — {len(self._df)} rows, {len(self._df.columns)} columns")
        return self
    
    def _build_standard(self) -> 'pd.DataFrame':
        import pandas as pd
        points = self._grid.to_points()
        print(f"Building standard dataset from {len(points)} points...")
        
        df = pd.DataFrame(points, columns=['lat', 'lon'])
        
        all_sources = []
        if self._label_source:
            all_sources.append(self._label_source)
        all_sources.extend(self._sources)
        
        for source in all_sources:
            source_df = source.fetch(points)
            if 'idx' in source_df.columns:
                source_df = source_df.drop(columns=['idx'])
            df = pd.concat([df, source_df], axis=1)
        
        return df
    
    def _build_homogeneous(self, threshold: float) -> 'pd.DataFrame':
        import pandas as pd
        import h3
        
        if self._label_source is None:
            raise ValueError("Homogeneous strategy requires set_label_source()")
        
        print(f"Building homogeneous dataset (threshold={threshold})...")
        
        if hasattr(self._grid, '_cells'):
            cells = list(self._grid._cells)
            homogeneity_df = self._label_source.fetch_homogeneity(cells)
            
            clean = homogeneity_df[homogeneity_df['homogeneity'] >= threshold]
            print(f"Kept {len(clean)}/{len(cells)} cells above threshold {threshold}")
            
            points = [h3.cell_to_latlng(cell) for cell in clean['h3_cell']]
            
        else:
            points, clean = self._fetch_kronecker_homogeneity(threshold)
        
        df = pd.DataFrame(points, columns=['lat', 'lon'])
        df['label'] = clean['label'].values
        df['label_name'] = clean['label_name'].values
        df['homogeneity'] = clean['homogeneity'].values
        
        for source in self._sources:
            source_df = source.fetch(points)
            if 'idx' in source_df.columns:
                source_df = source_df.drop(columns=['idx'])
            df = pd.concat([df.reset_index(drop=True), 
                          source_df.reset_index(drop=True)], axis=1)
        
        return df
    
    def _build_adaptive(
        self, 
        threshold: float, 
        max_depth: int, 
        checkpoint_dir: str = 'data/checkpoints',
        sampling: str = 'centroid'
    ) -> 'pd.DataFrame':
        
        if self._label_source is None:
            raise ValueError("Adaptive strategy requires set_label_source()")
        if not hasattr(self._grid, '_cells'):
            raise ValueError("Adaptive strategy requires H3GridBuilder")
        
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        print(f"Building adaptive dataset (threshold={threshold}, max_depth={max_depth}, sampling={sampling})...")
        
        all_clean_cells = []
        cells_to_process = list(self._grid._cells)
        current_res = self._grid.resolution
        
        for depth in range(max_depth + 1):
            checkpoint_path = f"{checkpoint_dir}/depth_{depth}_res{current_res}.csv"
            
            if Path(checkpoint_path).exists():
                print(f"Depth {depth}: loading from checkpoint {checkpoint_path}")
                homogeneity_df = pd.read_csv(checkpoint_path)
                clean = homogeneity_df[homogeneity_df['homogeneity'] >= threshold]
                noisy = homogeneity_df[homogeneity_df['homogeneity'] < threshold]
            else:
                print(f"Depth {depth}: processing {len(cells_to_process)} cells at res {current_res}")
                homogeneity_df = self._label_source.fetch_homogeneity(cells_to_process)
                homogeneity_df.to_csv(checkpoint_path, index=False)
                print(f"  Checkpoint saved: {checkpoint_path}")
                clean = homogeneity_df[homogeneity_df['homogeneity'] >= threshold]
                noisy = homogeneity_df[homogeneity_df['homogeneity'] < threshold]
            
            print(f"  Clean: {len(clean)}, Noisy: {len(noisy)}")
            all_clean_cells.append(clean)
            
            if depth == max_depth or len(noisy) == 0:
                break
            
            new_cells = []
            for cell in noisy['h3_cell']:
                new_cells.extend(h3.cell_to_children(cell, current_res + 1))
            
            cells_to_process = new_cells
            current_res += 1
        
        all_clean = pd.concat(all_clean_cells, ignore_index=True)
        print(f"Total clean cells: {len(all_clean)}")
        
        if sampling == 'centroid':
            points = [h3.cell_to_latlng(cell) for cell in all_clean['h3_cell']]
            labels_df = all_clean[['h3_cell', 'label', 'label_name', 'homogeneity']].copy()
            labels_df['sample_cell'] = labels_df['h3_cell']
            
        elif sampling == 'children':
            rows = []
            for _, row in all_clean.iterrows():
                parent_res = h3.get_resolution(row['h3_cell'])
                children = h3.cell_to_children(row['h3_cell'], parent_res + 1)
                for child in children:
                    rows.append({
                        'h3_cell': row['h3_cell'],
                        'sample_cell': child,
                        'label': row['label'],
                        'label_name': row['label_name'],
                        'homogeneity': row['homogeneity']
                    })
            labels_df = pd.DataFrame(rows)
            points = [h3.cell_to_latlng(cell) for cell in labels_df['sample_cell']]
            print(f"Expanded to {len(points)} points (7 children per clean cell)")
        
        else:
            raise ValueError(f"Unknown sampling: {sampling}. Use 'centroid' or 'children'")
        
        df = pd.DataFrame(points, columns=['lat', 'lon'])
        df['label'] = labels_df['label'].values
        df['label_name'] = labels_df['label_name'].values
        df['homogeneity'] = labels_df['homogeneity'].values
        df['h3_cell'] = labels_df['h3_cell'].values
        df['sample_cell'] = labels_df['sample_cell'].values
        
        for source in self._sources:
            source_df = source.fetch(points)
            if 'idx' in source_df.columns:
                source_df = source_df.drop(columns=['idx'])
            df = pd.concat([df.reset_index(drop=True),
                        source_df.reset_index(drop=True)], axis=1)
        
        return df
    
    def _fetch_kronecker_homogeneity(self, threshold: float):
        """For Kronecker points - check homogeneity using 500m buffer."""
        import ee
        import pandas as pd
        
        points = self._grid.to_points()
        image = self._label_source._get_image()
        clean_points = []
        clean_rows = []
        
        for lat, lon in tqdm(points, desc="Checking Kronecker homogeneity"):
            point = ee.Geometry.Point(lon, lat)
            region = point.buffer(500)
            
            counts = image.reduceRegion(
                reducer=ee.Reducer.frequencyHistogram(),
                geometry=region,
                scale=10,
                maxPixels=1e6
            ).getInfo()
            
            histogram = counts.get('Map', {})
            if not histogram:
                continue
            
            total = sum(histogram.values())
            majority_class = max(histogram, key=histogram.get)
            homogeneity = histogram[majority_class] / total
            
            if homogeneity >= threshold:
                clean_points.append((lat, lon))
                clean_rows.append({
                    'label': int(majority_class),
                    'label_name': self._label_source.CLASS_NAMES.get(int(majority_class)),
                    'homogeneity': homogeneity
                })
        
        return clean_points, pd.DataFrame(clean_rows)
    
    def save(self, path: str, format: str = 'csv') -> 'DatasetBuilder':
        if self._df is None:
            raise ValueError("No data — call build() first")
        
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'parquet':
            self._df.to_parquet(path, index=False, engine='fastparquet')
        elif format == 'csv':
            self._df.to_csv(path, index=False)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        print(f"Saved to {path}")
        return self
    
    @property
    def data(self) -> 'pd.DataFrame':
        if self._df is None:
            raise ValueError("No data — call build() first")
        return self._df
    
    def __repr__(self):
        sources = [str(s) for s in self._sources]
        return f"DatasetBuilder(grid={self._grid}, label_source={self._label_source}, sources={sources})"