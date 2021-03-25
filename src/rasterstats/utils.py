# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
import sys
import numpy as np
from rasterio import features
from affine import Affine
from numpy import min_scalar_type
from shapely.geometry import box, MultiPolygon
from .io import window_bounds


DEFAULT_STATS = ['count', 'min', 'max', 'mean']
VALID_STATS = DEFAULT_STATS + \
    ['sum', 'std', 'median', 'majority', 'minority', 'unique', 'range', 'nodata', 'nan']
#  also percentile_{q} but that is handled as special case


def get_percentile(stat):
    if not stat.startswith('percentile_'):
        raise ValueError("must start with 'percentile_'")
    qstr = stat.replace("percentile_", '')
    q = float(qstr)
    if q > 100.0:
        raise ValueError('percentiles must be <= 100')
    if q < 0.0:
        raise ValueError('percentiles must be >= 0')
    return q


def rasterize_geom(geom, like, all_touched=False):
    """
    Parameters
    ----------
    geom: GeoJSON geometry
    like: raster object with desired shape and transform
    all_touched: rasterization strategy

    Returns
    -------
    ndarray: boolean
    """
    geoms = [(geom, 1)]
    rv_array = features.rasterize(
        geoms,
        out_shape=like.shape,
        transform=like.affine,
        fill=0,
        dtype='uint8',
        all_touched=all_touched)
    return rv_array.astype(bool)


# https://stackoverflow.com/questions/8090229/
#   resize-with-averaging-or-rebin-a-numpy-2d-array/8090605#8090605
def rebin_sum(a, shape, dtype):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).sum(-1, dtype=dtype).sum(1, dtype=dtype)


class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

def rasterize_pctcover_geom(geom, like, scale=None, all_touched=False):
    """
    Parameters
    ----------
    geom: GeoJSON geometry
    like: raster object with desired shape and transform
    scale: scale at which to generate percent cover estimate

    Returns
    -------
    ndarray: float32
    """
    scale = scale if scale is not None else 10
    min_dtype = min_scalar_type(scale**2)

    pixel_size_lon = like.affine[0]/scale
    pixel_size_lat = like.affine[4]/scale

    topleftlon = like.affine[2]
    topleftlat = like.affine[5]

    new_affine = Affine(pixel_size_lon, 0, topleftlon,
                        0, pixel_size_lat, topleftlat)

    new_shape = (like.shape[0]*scale, like.shape[1]*scale)

    new_like = objectview({'shape': new_shape, 'affine': new_affine})

    rv_array = rasterize_geom(geom, new_like, all_touched=all_touched)
    rv_array = rebin_sum(rv_array, like.shape, min_dtype)

    return rv_array.astype('float32') / (scale**2)


def stats_to_csv(stats):
    if sys.version_info[0] >= 3:
        from io import StringIO as IO  # pragma: no cover
    else:
        from cStringIO import StringIO as IO  # pragma: no cover

    import csv

    csv_fh = IO()

    keys = set()
    for stat in stats:
        for key in list(stat.keys()):
            keys.add(key)

    fieldnames = sorted(list(keys), key=str)

    csvwriter = csv.DictWriter(csv_fh, delimiter=str(","), fieldnames=fieldnames)
    csvwriter.writerow(dict((fn, fn) for fn in fieldnames))
    for row in stats:
        csvwriter.writerow(row)
    contents = csv_fh.getvalue()
    csv_fh.close()
    return contents


def check_stats(stats, categorical):
    if not stats:
        if not categorical:
            stats = DEFAULT_STATS
        else:
            stats = []
    else:
        if isinstance(stats, str):
            if stats in ['*', 'ALL']:
                stats = VALID_STATS
            else:
                stats = stats.split()
    for x in stats:
        if x.startswith("percentile_"):
            get_percentile(x)
        elif x not in VALID_STATS:
            raise ValueError(
                "Stat `%s` not valid; "
                "must be one of \n %r" % (x, VALID_STATS))

    run_count = False
    if categorical or 'majority' in stats or 'minority' in stats or 'unique' in stats:
        # run the counter once, only if needed
        run_count = True

    return stats, run_count


def remap_categories(category_map, stats):
    def lookup(m, k):
        """ Dict lookup but returns original key if not found
        """
        try:
            return m[k]
        except KeyError:
            return k

    return {lookup(category_map, k): v
            for k, v in stats.items()}


def key_assoc_val(d, func, exclude=None):
    """return the key associated with the value returned by func
    """
    vs = list(d.values())
    ks = list(d.keys())
    key = ks[vs.index(func(vs))]
    return key


def boxify_points(geom, rast):
    """
    Point and MultiPoint don't play well with GDALRasterize
    convert them into box polygons 99% cellsize, centered on the raster cell
    """
    if 'Point' not in geom.type:
        raise ValueError("Points or multipoints only")

    buff = -0.01 * abs(min(rast.affine.a, rast.affine.e))

    if geom.type == 'Point':
        pts = [geom]
    elif geom.type == "MultiPoint":
        pts = geom.geoms
    geoms = []
    for pt in pts:
        row, col = rast.index(pt.x, pt.y)
        win = ((row, row + 1), (col, col + 1))
        geoms.append(box(*window_bounds(win, rast.affine)).buffer(buff))

    return MultiPolygon(geoms)



def rs_mean(masked, cover_weights=None):
    if cover_weights is not None:
        val = float(
            np.sum(masked * cover_weights) /
            np.sum(~masked.mask * cover_weights))
    else:
        val = float(masked.mean())
    return val


def rs_count(masked, cover_weights=None):
    if cover_weights is not None:
        val = float(np.sum(~masked.mask * cover_weights))
    else:
        val = int(masked.count())
    return val


def rs_sum(masked, cover_weights=None):
    if cover_weights is not None:
        val = float(np.sum(masked * cover_weights))
    else:
        val = float(masked.sum())
    return val
