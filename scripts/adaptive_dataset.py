import ee
import sys
import importlib
import geopandas as gpd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import utilities
importlib.reload(utilities)

from utilities import (
    DatasetBuilder, 
    H3GridBuilder, 
    AlphaEarth, 
    ESAWorldCover
)

ee.Initialize(project="lucera-494618")

nuts = gpd.read_file("NUTS_RG_01M_2021_4326.gpkg")
country_bbox = nuts[nuts['CNTR_CODE'] == 'PL'].union_all()

pl2 = nuts[nuts['NUTS_ID'].isin(['PL21', 'PL22'])].union_all()
PL2_BBOX = pl2.bounds

H3_FILENAME = 'h3_pl_worldcover_2023'

h3_grid = H3GridBuilder(bbox=country_bbox.bounds, resolution=7)

dataset_adaptive = (
    DatasetBuilder()
    .from_grid(H3GridBuilder(bbox=PL2_BBOX, resolution=7))
    .set_label_source(ESAWorldCover())
    .add_source(AlphaEarth(year=2023))
    .build(strategy='adaptive', threshold=0.8, max_depth=2, checkpoint_dir='data/pl2_checkpoints')
    .save("data/pl2_adaptive.csv")
)