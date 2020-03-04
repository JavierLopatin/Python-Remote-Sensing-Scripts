import sys
!{sys.executable} -m pip install geebap

import geebap
from geetools import tools

import pprint
pp = pprint.PrettyPrinter(indent=2)

import ee
ee.Initialize()

# COLLECTIONS
# col_group = satcol.ColGroup.Landsat()

# SEASON
a_season = geebap.season.Season.Growing_South()

# MASKS
cld_mask = masks.Clouds()
# equiv_mask = masks.Equivalent()  # DEPRECATED

# Combine masks in a tuple
masks = (cld_mask,)

# FILTERS
filt_cld = filters.CloudsPercent()
filt_mask = filters.MaskPercent()

# Combine filters in a tuple
filters = (filt_cld, filt_mask)

# SCORES
best_doy = scores.Doy()
sat = scores.Satellite()
op = scores.AtmosOpacity()
out = scores.Outliers(("ndvi",))
ind = scores.Index("ndvi")
mascpor = scores.MaskPercent()
dist = scores.CloudDist()

# Combine scores in a tuple
scores = (best_doy, sat, op, out, ind, mascpor, dist)

# BAP OBJECT
bap = bap.Bap(year=2010, range=(0, 0),
              season=a_season,
              # colgroup=col_group,  # if colgroup is None, it'll use season.SeasonPriority
              masks=masks,
              scores=scores,
              filters=filters)

# SITE
site = ee.Geometry.Polygon([[-71,-42],
                            [-71,-43],
                            [-72,-43],
                            [-72,-42]])

# COMPOSITE
composite = bap.bestpixel(site=site,
                          indices=("ndvi",))

# The result (composite) is a namedtuple, so
image = composite.image

# image is a ee.Image object, so you can do anything
# from here..

one_value = tools.image.get_value(image,
                            site.centroid(),
                            30, 'client')

pp.pprint(one_value)
