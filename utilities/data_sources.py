from abc import ABC, abstractmethod
import ee
import pandas as pd
from tqdm import tqdm
import time


class EarthEngineSource(ABC):
    """Abstract base for all EE data sources."""
    
    @abstractmethod
    def fetch(self, points: list[tuple[float, float]]) -> pd.DataFrame:
        """
        Fetch data for list of (lat, lon) points.
        Returns DataFrame with one row per point.
        """
        pass
    
    def _points_to_fc(self, points: list[tuple[float, float]]) -> ee.FeatureCollection:
        """Convert (lat, lon) list to EE FeatureCollection."""
        features = [
            ee.Feature(ee.Geometry.Point(lon, lat), {'idx': i})
            for i, (lat, lon) in enumerate(points)
        ]
        return ee.FeatureCollection(features)
    
class AlphaEarth(EarthEngineSource):
    """
    Google Alpha Earth embeddings via Earth Engine.
    Asset: GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL
    Returns 64-dimensional embedding per point.
    """
    
    COLLECTION = 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL'
    
    def __init__(self, year: int = 2023, chunk_size: int = 500):
        self.year = year
        self.chunk_size = chunk_size
    
    def _get_image(self, fc: ee.FeatureCollection) -> ee.Image:
        return (
            ee.ImageCollection(self.COLLECTION)
            .filterDate(f'{self.year}-01-01', f'{self.year+1}-01-01')
            .filterBounds(fc)
            .mosaic()
        )
    
    def fetch(self, points: list[tuple[float, float]]) -> pd.DataFrame:
        all_rows = []
        chunks = [points[i:i+self.chunk_size] for i in range(0, len(points), self.chunk_size)]
        
        for chunk in tqdm(chunks, desc="Fetching AlphaEarth embeddings"):
            fc = self._points_to_fc(chunk)
            image = self._get_image(fc)
            
            sampled = image.sampleRegions(
                collection=fc,
                scale=10,
                projection=image.projection(),
                geometries=False
            )
            
            result = sampled.getInfo()
            rows = [f['properties'] for f in result['features']]
            all_rows.extend(rows)
        
        df = pd.DataFrame(all_rows)
        
        return df
    
    def __repr__(self):
        return f"AlphaEarth(year={self.year})"
    
class ESAWorldCover(EarthEngineSource):
    """
    ESA WorldCover land cover labels via Earth Engine.
    Asset: ESA/WorldCover/v200
    Returns single class label per point (10, 20, 30... etc.)
    
    Classes:
        10: Tree cover
        20: Shrubland
        30: Grassland
        40: Cropland
        50: Built-up
        60: Bare/sparse vegetation
        70: Snow and ice
        80: Permanent water bodies
        90: Herbaceous wetland
        95: Mangroves
        100: Moss and lichen
    """
    
    COLLECTION = 'ESA/WorldCover/v200'
    CLASS_NAMES = {
        10: 'Tree cover',
        20: 'Shrubland',
        30: 'Grassland',
        40: 'Cropland',
        50: 'Built-up',
        60: 'Bare/sparse vegetation',
        70: 'Snow and ice',
        80: 'Permanent water bodies',
        90: 'Herbaceous wetland',
        95: 'Mangroves',
        100: 'Moss and lichen'
    }
    
    def __init__(self, chunk_size: int = 500):
        self.chunk_size = chunk_size
    
    def _get_image(self) -> 'ee.Image':
        import ee
        return ee.ImageCollection(self.COLLECTION).first()
    
    def fetch(self, points: list[tuple[float, float]]) -> 'pd.DataFrame':
        import ee
        import pandas as pd
        all_rows = []
        chunks = [points[i:i+self.chunk_size] for i in range(0, len(points), self.chunk_size)]
        image = self._get_image()
        
        for chunk in tqdm(chunks, desc="Fetching ESA WorldCover labels"):
            fc = self._points_to_fc(chunk)
            sampled = image.sampleRegions(
                collection=fc,
                scale=10,
                projection=image.projection(),
                geometries=False
            )
            result = sampled.getInfo()
            rows = [f['properties'] for f in result['features']]
            all_rows.extend(rows)
        
        df = pd.DataFrame(all_rows)
        if 'Map' in df.columns:
            df = df.rename(columns={'Map': 'label'})
            df['label_name'] = df['label'].map(self.CLASS_NAMES)
        return df

    
    def fetch_homogeneity(
        self, 
        cells: list[str],
        chunk_size: int = 20
    ) -> 'pd.DataFrame':
        """
        For each H3 cell, compute majority class and homogeneity score.
        Returns DataFrame with columns: h3_cell, label, label_name, homogeneity
        """
        import ee
        import pandas as pd
        import h3
        
        image = self._get_image()
        all_rows = []
        chunks = [cells[i:i+chunk_size] for i in range(0, len(cells), chunk_size)]
        
        for chunk in tqdm(chunks, desc="Computing homogeneity"):
            features = []
            for cell in chunk:
                boundary = h3.cell_to_boundary(cell)
                coords = [[lon, lat] for lat, lon in boundary]
                polygon = ee.Geometry.Polygon(coords)
                
                # get pixel count per class within hex
                counts = image.reduceRegion(
                    reducer=ee.Reducer.frequencyHistogram(),
                    geometry=polygon,
                    scale=10,
                    maxPixels=1e6
                )
                
                features.append(
                    ee.Feature(None, {
                        'h3_cell': cell,
                        'histogram': counts.get('Map')
                    })
                )
            
            fc = ee.FeatureCollection(features)
            result = fc.getInfo()
            
            for f in result['features']:
                props = f['properties']
                cell = props['h3_cell']
                histogram = props.get('histogram', {})
                
                if not histogram:
                    continue
                
                total = sum(histogram.values())
                majority_class = max(histogram, key=histogram.get)
                majority_count = histogram[majority_class]
                homogeneity = majority_count / total if total > 0 else 0
                
                all_rows.append({
                    'h3_cell': cell,
                    'label': int(majority_class),
                    'label_name': self.CLASS_NAMES.get(int(majority_class), 'Unknown'),
                    'homogeneity': homogeneity,
                    'total_pixels': total
                })
        
        return pd.DataFrame(all_rows)
    
    def __repr__(self):
        return "ESAWorldCover()"