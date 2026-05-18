from abc import ABC, abstractmethod
import h3
from shapely.geometry import Point, Polygon
from typing import Optional
import numpy as np
from enum import Enum
from scipy.stats import qmc

class SamplingMethod(Enum):
    CENTROID = "centroid"
    RANDOM = "random"
    STRATIFIED = "stratified" 


class GridBuilder(ABC):
    @abstractmethod
    def to_points(
        self, 
        method: SamplingMethod = SamplingMethod.CENTROID,
        n_per_cell: int = 1,
        seed: int = 42
    ) -> list[tuple[float, float]]:
        """
        Return list of (lat, lon) points.
        
        Args:
            method: sampling strategy
            n_per_cell: points per cell (ignored for CENTROID)
            seed: random seed for RANDOM method
        """
        pass
    
    @abstractmethod
    def refine(self, resolution: int) -> 'GridBuilder':
        """Subdivide cells to higher resolution."""
        pass
    
    def filter(self, mask_fn) -> 'GridBuilder':
        """Filter cells by arbitrary function — override if needed."""
        raise NotImplementedError
    
    def show_points(
        self, 
        zoom: int = 6,
        color: str = 'orange',
        method: SamplingMethod = SamplingMethod.CENTROID,
        n_per_cell: int = 1
    ):
        """Show grid points as dots on folium map. Works for all grid types."""
        try:
            import folium
        except ImportError:
            raise ImportError("pip install folium")
        
        points = self.to_points(method=method, n_per_cell=n_per_cell)
        center = [
            sum(p[0] for p in points) / len(points),
            sum(p[1] for p in points) / len(points)
        ]
        
        m = folium.Map(location=center, zoom_start=zoom)
        for lat, lon in points:
            folium.CircleMarker(
                location=[lat, lon],
                radius=2,
                color=color,
                fill=True,
                fill_opacity=0.7,
                weight=0
            ).add_to(m)
        return m
    
    
class H3GridBuilder(GridBuilder):
    """
    Hexagonal grid builder using H3.
    
    Usage:
        grid = H3GridBuilder(bbox=(17.8, 49.2, 22.9, 50.6), resolution=7)
        grid.refine(8).to_points()
    """
    
    def __init__(
        self, 
        bbox: Optional[tuple] = None,
        polygon: Optional[dict] = None,
        resolution: int = 7
    ):
        """
        Args:
            bbox: (min_lon, min_lat, max_lon, max_lat)
            polygon: GeoJSON-like dict or shapely geometry
            resolution: H3 resolution (0=coarsest, 15=finest)
        """
        self.resolution = resolution
        self._cells = self._init_cells(bbox, polygon)
    
    def _init_cells(self, bbox, polygon) -> set:
        if bbox is not None:
            min_lon, min_lat, max_lon, max_lat = bbox
            geojson = {
                "type": "Polygon",
                "coordinates": [[
                    [min_lon, min_lat],
                    [max_lon, min_lat],
                    [max_lon, max_lat],
                    [min_lon, max_lat],
                    [min_lon, min_lat]
                ]]
            }
        elif polygon is not None:
            if hasattr(polygon, '__geo_interface__'):
                geojson = polygon.__geo_interface__
            else:
                geojson = polygon
        else:
            raise ValueError("Either bbox or polygon must be provided")
        
        return set(h3.geo_to_cells(geojson, self.resolution))
    
    def refine(self, resolution: int) -> 'H3GridBuilder':
        """Subdivide all cells to higher resolution."""
        if resolution <= self.resolution:
            raise ValueError(f"New resolution {resolution} must be higher than current {self.resolution}")
        
        new_cells = set()
        for cell in self._cells:
            new_cells.update(h3.cell_to_children(cell, resolution))
        
        new_grid = H3GridBuilder.__new__(H3GridBuilder)
        new_grid.resolution = resolution
        new_grid._cells = new_cells
        return new_grid
    
    def filter(self, mask_fn) -> 'H3GridBuilder':
        """Filter cells by function(lat, lon) -> bool."""
        filtered = set()
        for cell in self._cells:
            lat, lon = h3.cell_to_latlng(cell)
            if mask_fn(lat, lon):
                filtered.add(cell)
        
        new_grid = H3GridBuilder.__new__(H3GridBuilder)
        new_grid.resolution = self.resolution
        new_grid._cells = filtered
        return new_grid
    
    def to_points(
        self,
        method: SamplingMethod = SamplingMethod.CENTROID,
        n_per_cell: int = 1,
        seed: int = 42
    ) -> list[tuple[float, float]]:
        
        if method == SamplingMethod.CENTROID:
            return [h3.cell_to_latlng(cell) for cell in self._cells]
        
        elif method == SamplingMethod.RANDOM:
            np.random.seed(seed)
            points = []
            for cell in self._cells:
                boundary = h3.cell_to_boundary(cell)
                poly = Polygon([(lon, lat) for lat, lon in boundary])
                bounds = poly.bounds
                count = 0
                attempts = 0
                while count < n_per_cell and attempts < 100:
                    x = np.random.uniform(bounds[0], bounds[2])
                    y = np.random.uniform(bounds[1], bounds[3])
                    pt = Point(x, y)
                    if poly.contains(pt):
                        points.append((pt.y, pt.x))
                        count += 1
                    attempts += 1
            return points
        
        elif method == SamplingMethod.STRATIFIED:
            next_res = self.resolution + 1
            sub_points = []
            for cell in self._cells:
                children = h3.cell_to_children(cell, next_res)
                sub_points.extend([h3.cell_to_latlng(c) for c in children])
            return sub_points
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
    def __len__(self):
        return len(self._cells)
    
    def __repr__(self):
        return f"H3GridBuilder(resolution={self.resolution}, cells={len(self._cells)})"
    
    def show(self, zoom: int = 6) -> None:
        """Render grid on interactive folium map."""
        try:
            import folium
        except ImportError:
            raise ImportError("pip install folium")
        
        points = self.to_points()
        center = [
            sum(p[0] for p in points) / len(points),
            sum(p[1] for p in points) / len(points)
        ]
        
        m = folium.Map(location=center, zoom_start=zoom)
        
        for cell in self._cells:
            boundary = h3.cell_to_boundary(cell)
            
            folium.Polygon(
                locations=boundary,
                color='orange',
                fill=True,
                fill_opacity=0.1,
                weight=1
            ).add_to(m)
        
        return m
    

class KroneckerGridBuilder(GridBuilder):
    """
    Quasi-random grid using R2/Kronecker sequence.
    Better spatial uniformity than random, no rigid grid structure.
    """
    
    def __init__(
        self,
        bbox: Optional[tuple] = None,
        polygon: Optional[dict] = None,
        n_points: int = 1000,
        seed: float = 0.5
    ):
        self.bbox = bbox
        self.polygon = polygon
        self.n_points = n_points
        self.seed = seed
        self._points = self._init_points()
    
    def _init_points(self) -> list[tuple[float, float]]:
        if self.bbox is None:
            raise ValueError("bbox required for now")
        
        min_lon, min_lat, max_lon, max_lat = self.bbox
        
        n_generate = self.n_points * 3 if self.polygon else self.n_points
        
        sampler = qmc.Halton(d=2, scramble=True, seed=42)
        samples = sampler.random(n=n_generate)
        
        lons = min_lon + samples[:, 0] * (max_lon - min_lon)
        lats = min_lat + samples[:, 1] * (max_lat - min_lat)
        
        points = []
        for lat, lon in zip(lats, lons):
            if self.polygon is not None:
                if not self.polygon.contains(Point(lon, lat)):
                    continue
            points.append((lat, lon))
            if len(points) >= self.n_points:
                break
        
        return points
    
    def to_points(
        self,
        method: SamplingMethod = SamplingMethod.CENTROID,
        n_per_cell: int = 1,
        seed: int = 42
    ) -> list[tuple[float, float]]:
        return self._points
    
    def refine(self, resolution: int = None) -> 'KroneckerGridBuilder':
        """Double the number of points - R2 equivalent of refining."""
        new_grid = KroneckerGridBuilder(
            bbox=self.bbox,
            polygon=self.polygon,
            n_points=self.n_points * 7, 
            seed=self.seed
        )
        return new_grid
    
    def filter(self, mask_fn) -> 'KroneckerGridBuilder':
        filtered = [(lat, lon) for lat, lon in self._points if mask_fn(lat, lon)]
        new_grid = KroneckerGridBuilder.__new__(KroneckerGridBuilder)
        new_grid.bbox = self.bbox
        new_grid.polygon = self.polygon
        new_grid.n_points = len(filtered)
        new_grid.seed = self.seed
        new_grid._points = filtered
        return new_grid
    
    def show(self, zoom: int = 6) -> None:
        import folium
        center = [
            sum(p[0] for p in self._points) / len(self._points),
            sum(p[1] for p in self._points) / len(self._points)
        ]
        m = folium.Map(location=center, zoom_start=zoom)
        for lat, lon in self._points:
            folium.CircleMarker(
                location=[lat, lon],
                radius=2,
                color='blue',
                fill=True,
                fill_opacity=0.7,
                weight=0
            ).add_to(m)
        return m
    
    def __len__(self):
        return len(self._points)
    
    def __repr__(self):
        return f"KroneckerGridBuilder(n_points={self.n_points})"