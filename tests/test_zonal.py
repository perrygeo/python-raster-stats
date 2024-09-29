# test zonal stats
import json
import os
import sys

import numpy as np
import geopandas as gpd
import pytest
import rasterio
import simplejson
from affine import Affine
from shapely.geometry import Polygon

from rasterstats import raster_stats, zonal_stats, gen_zonal_stats #, gdf_zonal_stats
from rasterstats.io import read_featurecollection, read_features
from rasterstats.utils import VALID_STATS

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
raster = os.path.join(DATA, "slope.tif")


def test_main():
    polygons = os.path.join(DATA, "polygons.shp")
    stats = zonal_stats(polygons, raster)
    for key in ["count", "min", "max", "mean"]:
        assert key in stats[0]
    assert len(stats) == 2
    assert stats[0]["count"] == 75
    assert stats[1]["count"] == 50
    assert round(stats[0]["mean"], 2) == 14.66


# remove after band_num alias is removed
def test_band_alias():
    polygons = os.path.join(DATA, "polygons.shp")
    stats_a = zonal_stats(polygons, raster)
    stats_b = zonal_stats(polygons, raster, band=1)
    with pytest.deprecated_call():
        stats_c = zonal_stats(polygons, raster, band_num=1)
    assert stats_a[0]["count"] == stats_b[0]["count"] == stats_c[0]["count"]


def test_zonal_global_extent():
    polygons = os.path.join(DATA, "polygons.shp")
    stats = zonal_stats(polygons, raster)
    global_stats = zonal_stats(polygons, raster, global_src_extent=True)
    assert stats == global_stats


def test_zonal_nodata():
    polygons = os.path.join(DATA, "polygons.shp")
    stats = zonal_stats(polygons, raster, nodata=0)
    assert len(stats) == 2
    assert stats[0]["count"] == 75
    assert stats[1]["count"] == 50


def test_doesnt_exist():
    nonexistent = os.path.join(DATA, "DOESNOTEXIST.shp")
    with pytest.raises(ValueError):
        zonal_stats(nonexistent, raster)


def test_nonsense():
    polygons = os.path.join(DATA, "polygons.shp")
    with pytest.raises(ValueError):
        zonal_stats("blaghrlargh", raster)
    with pytest.raises(OSError):
        zonal_stats(polygons, "blercherlerch")
    with pytest.raises(ValueError):
        zonal_stats(
            [
                "blaghrlargh",
            ],
            raster,
        )


# Different geometry types
def test_points():
    points = os.path.join(DATA, "points.shp")
    stats = zonal_stats(points, raster)
    # three features
    assert len(stats) == 3
    # three pixels
    assert sum([x["count"] for x in stats]) == 3
    assert round(stats[0]["mean"], 3) == 11.386
    assert round(stats[1]["mean"], 3) == 35.547


def test_points_categorical():
    points = os.path.join(DATA, "points.shp")
    categorical_raster = os.path.join(DATA, "slope_classes.tif")
    stats = zonal_stats(points, categorical_raster, categorical=True)
    # three features
    assert len(stats) == 3
    assert "mean" not in stats[0]
    assert stats[0][1.0] == 1
    assert stats[1][2.0] == 1


def test_lines():
    lines = os.path.join(DATA, "lines.shp")
    stats = zonal_stats(lines, raster)
    assert len(stats) == 2
    assert stats[0]["count"] == 58
    assert stats[1]["count"] == 32


# Test multigeoms
def test_multipolygons():
    multipolygons = os.path.join(DATA, "multipolygons.shp")
    stats = zonal_stats(multipolygons, raster)
    assert len(stats) == 1
    assert stats[0]["count"] == 125


def test_multilines():
    multilines = os.path.join(DATA, "multilines.shp")
    stats = zonal_stats(multilines, raster)
    assert len(stats) == 1
    # can differ slightly based on platform/gdal version
    assert stats[0]["count"] in [89, 90]


def test_multipoints():
    multipoints = os.path.join(DATA, "multipoints.shp")
    stats = zonal_stats(multipoints, raster)
    assert len(stats) == 1
    assert stats[0]["count"] == 3


def test_categorical():
    polygons = os.path.join(DATA, "polygons.shp")
    categorical_raster = os.path.join(DATA, "slope_classes.tif")
    stats = zonal_stats(polygons, categorical_raster, categorical=True)
    assert len(stats) == 2
    assert stats[0][1.0] == 75
    assert 5.0 in stats[1]


def test_categorical_map():
    polygons = os.path.join(DATA, "polygons.shp")
    categorical_raster = os.path.join(DATA, "slope_classes.tif")
    catmap = {5.0: "cat5"}
    stats = zonal_stats(
        polygons, categorical_raster, categorical=True, category_map=catmap
    )
    assert len(stats) == 2
    assert stats[0][1.0] == 75
    assert 5.0 not in stats[1]
    assert "cat5" in stats[1]


def test_specify_stats_list():
    polygons = os.path.join(DATA, "polygons.shp")
    stats = zonal_stats(polygons, raster, stats=["min", "max"])
    assert sorted(stats[0].keys()) == sorted(["min", "max"])
    assert "count" not in list(stats[0].keys())


def test_specify_all_stats():
    polygons = os.path.join(DATA, "polygons.shp")
    stats = zonal_stats(polygons, raster, stats="ALL")
    assert sorted(stats[0].keys()) == sorted(VALID_STATS)
    stats = zonal_stats(polygons, raster, stats="*")
    assert sorted(stats[0].keys()) == sorted(VALID_STATS)


def test_specify_stats_string():
    polygons = os.path.join(DATA, "polygons.shp")
    stats = zonal_stats(polygons, raster, stats="min max")
    assert sorted(stats[0].keys()) == sorted(["min", "max"])
    assert "count" not in list(stats[0].keys())


def test_specify_stats_invalid():
    polygons = os.path.join(DATA, "polygons.shp")
    with pytest.raises(ValueError):
        zonal_stats(polygons, raster, stats="foo max")


def test_optional_stats():
    polygons = os.path.join(DATA, "polygons.shp")
    stats = zonal_stats(polygons, raster, stats="min max sum majority median std")
    assert stats[0]["min"] <= stats[0]["median"] <= stats[0]["max"]


def test_range():
    polygons = os.path.join(DATA, "polygons.shp")
    stats = zonal_stats(polygons, raster, stats="range min max")
    for stat in stats:
        assert stat["range"] == stat["max"] - stat["min"]
    ranges = [x["range"] for x in stats]
    # without min/max specified
    stats = zonal_stats(polygons, raster, stats="range")
    assert "min" not in stats[0]
    assert ranges == [x["range"] for x in stats]


def test_nodata():
    polygons = os.path.join(DATA, "polygons.shp")
    categorical_raster = os.path.join(DATA, "slope_classes.tif")
    stats = zonal_stats(
        polygons, categorical_raster, stats="*", categorical=True, nodata=1.0
    )
    assert stats[0]["majority"] is None
    assert stats[0]["count"] == 0  # no pixels; they're all null
    assert stats[1]["minority"] == 2.0
    assert stats[1]["count"] == 49  # used to be 50 if we allowed 1.0
    assert "1.0" not in stats[0]


def test_dataset_mask():
    polygons = os.path.join(DATA, "polygons.shp")
    raster = os.path.join(DATA, "dataset_mask.tif")
    stats = zonal_stats(polygons, raster, stats="*")
    assert stats[0]["count"] == 75
    assert stats[1]["count"] == 0


def test_partial_overlap():
    polygons = os.path.join(DATA, "polygons_partial_overlap.shp")
    stats = zonal_stats(polygons, raster, stats="count")
    for res in stats:
        # each polygon should have at least a few pixels overlap
        assert res["count"] > 0


def test_no_overlap():
    polygons = os.path.join(DATA, "polygons_no_overlap.shp")
    stats = zonal_stats(polygons, raster, stats="count")
    for res in stats:
        # no polygon should have any overlap
        assert res["count"] == 0


def test_all_touched():
    polygons = os.path.join(DATA, "polygons.shp")
    stats = zonal_stats(polygons, raster, all_touched=True)
    assert stats[0]["count"] == 95  # 75 if ALL_TOUCHED=False
    assert stats[1]["count"] == 73  # 50 if ALL_TOUCHED=False


def test_ndarray_without_affine():
    with rasterio.open(raster) as src:
        polygons = os.path.join(DATA, "polygons.shp")
        with pytest.raises(ValueError):
            zonal_stats(polygons, src.read(1))  # needs affine kwarg


def _assert_dict_eq(a, b):
    """Assert that dicts a and b similar within floating point precision"""
    err = 1e-5
    for k in set(a.keys()).union(set(b.keys())):
        if a[k] == b[k]:
            continue
        try:
            if abs(a[k] - b[k]) > err:
                raise AssertionError(f"{k}: {a[k]} != {b[k]}")
        except TypeError:  # can't take abs, nan
            raise AssertionError(f"{a[k]} != {b[k]}")


def test_ndarray():
    with rasterio.open(raster) as src:
        arr = src.read(1)
        affine = src.transform

    polygons = os.path.join(DATA, "polygons.shp")
    stats = zonal_stats(polygons, arr, affine=affine)
    stats2 = zonal_stats(polygons, raster)
    for s1, s2 in zip(stats, stats2):
        _assert_dict_eq(s1, s2)
    with pytest.raises(AssertionError):
        _assert_dict_eq(stats[0], stats[1])
    assert stats[0]["count"] == 75
    assert stats[1]["count"] == 50

    points = os.path.join(DATA, "points.shp")
    stats = zonal_stats(points, arr, affine=affine)
    assert stats == zonal_stats(points, raster)
    assert sum([x["count"] for x in stats]) == 3
    assert round(stats[0]["mean"], 3) == 11.386
    assert round(stats[1]["mean"], 3) == 35.547


def test_alias():
    polygons = os.path.join(DATA, "polygons.shp")
    stats = zonal_stats(polygons, raster)
    with pytest.deprecated_call():
        stats2 = raster_stats(polygons, raster)
    assert stats == stats2


def test_add_stats():
    polygons = os.path.join(DATA, "polygons.shp")

    def mymean(x):
        return np.ma.mean(x)

    stats = zonal_stats(polygons, raster, add_stats={"mymean": mymean})
    for i in range(len(stats)):
        assert stats[i]["mean"] == stats[i]["mymean"]


def test_add_stats_prop():
    polygons = os.path.join(DATA, "polygons.shp")

    def mymean_prop(x, prop):
        return np.ma.mean(x) * prop["id"]

    stats = zonal_stats(polygons, raster, add_stats={"mymean_prop": mymean_prop})
    for i in range(len(stats)):
        assert stats[i]["mymean_prop"] == stats[i]["mean"] * (i + 1)

def test_add_stats_prop_and_array():
    polygons = os.path.join(DATA, "polygons.shp")

    def mymean_prop_and_array(x, prop, rv_array):
        # confirm that the object exists and is accessible.
        assert rv_array is not None
        return np.ma.mean(x) * prop["id"]

    stats = zonal_stats(polygons, raster, add_stats={"mymean_prop_and_array": mymean_prop_and_array})
    for i in range(len(stats)):
        assert stats[i]["mymean_prop_and_array"] == stats[i]["mean"] * (i + 1)


def test_mini_raster():
    polygons = os.path.join(DATA, "polygons.shp")
    stats = zonal_stats(polygons, raster, raster_out=True)
    stats2 = zonal_stats(
        polygons,
        stats[0]["mini_raster_array"],
        raster_out=True,
        affine=stats[0]["mini_raster_affine"],
    )
    assert (
        stats[0]["mini_raster_array"] == stats2[0]["mini_raster_array"]
    ).sum() == stats[0]["count"]


def test_percentile_good():
    polygons = os.path.join(DATA, "polygons.shp")
    stats = zonal_stats(polygons, raster, stats="median percentile_50 percentile_90")
    assert "percentile_50" in stats[0].keys()
    assert "percentile_90" in stats[0].keys()
    assert stats[0]["percentile_50"] == stats[0]["median"]
    assert stats[0]["percentile_50"] <= stats[0]["percentile_90"]


def test_zone_func_has_return():
    def example_zone_func(zone_arr):
        return np.ma.masked_array(np.full(zone_arr.shape, 1))

    polygons = os.path.join(DATA, "polygons.shp")
    stats = zonal_stats(polygons, raster, zone_func=example_zone_func)
    assert stats[0]["max"] == 1
    assert stats[0]["min"] == 1
    assert stats[0]["mean"] == 1


def test_zone_func_good():
    def example_zone_func(zone_arr):
        zone_arr[:] = 0

    polygons = os.path.join(DATA, "polygons.shp")
    stats = zonal_stats(polygons, raster, zone_func=example_zone_func)
    assert stats[0]["max"] == 0
    assert stats[0]["min"] == 0
    assert stats[0]["mean"] == 0


def test_zone_func_bad():
    not_a_func = "jar jar binks"
    polygons = os.path.join(DATA, "polygons.shp")
    with pytest.raises(TypeError):
        zonal_stats(polygons, raster, zone_func=not_a_func)


def test_percentile_nodata():
    polygons = os.path.join(DATA, "polygons.shp")
    categorical_raster = os.path.join(DATA, "slope_classes.tif")
    # By setting nodata to 1, one of our polygons is within the raster extent
    # but has an empty masked array
    stats = zonal_stats(polygons, categorical_raster, stats=["percentile_90"], nodata=1)
    assert "percentile_90" in stats[0].keys()
    assert [None, 5.0] == [x["percentile_90"] for x in stats]


def test_percentile_bad():
    polygons = os.path.join(DATA, "polygons.shp")
    with pytest.raises(ValueError):
        zonal_stats(polygons, raster, stats="percentile_101")


def test_json_serializable():
    polygons = os.path.join(DATA, "polygons.shp")
    stats = zonal_stats(
        polygons, raster, stats=VALID_STATS + ["percentile_90"], categorical=True
    )
    try:
        json.dumps(stats)
        simplejson.dumps(stats)
    except TypeError:
        pytest.fail("zonal_stats returned a list that wasn't JSON-serializable")


def test_direct_features_collections():
    polygons = os.path.join(DATA, "polygons.shp")
    features = read_features(polygons)
    collection = read_featurecollection(polygons)

    stats_direct = zonal_stats(polygons, raster)
    stats_features = zonal_stats(features, raster)
    stats_collection = zonal_stats(collection, raster)

    assert stats_direct == stats_features == stats_collection


def test_all_nodata():
    polygons = os.path.join(DATA, "polygons.shp")
    raster = os.path.join(DATA, "all_nodata.tif")
    stats = zonal_stats(polygons, raster, stats=["nodata", "count"])
    assert stats[0]["nodata"] == 75
    assert stats[0]["count"] == 0
    assert stats[1]["nodata"] == 50
    assert stats[1]["count"] == 0


def test_some_nodata():
    polygons = os.path.join(DATA, "polygons.shp")
    raster = os.path.join(DATA, "slope_nodata.tif")
    stats = zonal_stats(polygons, raster, stats=["nodata", "count"])
    assert stats[0]["nodata"] == 36
    assert stats[0]["count"] == 39
    assert stats[1]["nodata"] == 19
    assert stats[1]["count"] == 31


# update this if nan end up being incorporated into nodata
def test_nan_nodata():
    polygon = Polygon([[0, 0], [2, 0], [2, 2], [0, 2]])
    arr = np.array([[np.nan, 12.25], [-999, 12.75]])
    affine = Affine(1, 0, 0, 0, -1, 2)

    stats = zonal_stats(
        polygon, arr, affine=affine, nodata=-999, stats="nodata count sum mean min max"
    )

    assert stats[0]["nodata"] == 1
    assert stats[0]["count"] == 2
    assert stats[0]["mean"] == 12.5
    assert stats[0]["min"] == 12.25
    assert stats[0]["max"] == 12.75


def test_some_nodata_ndarray():
    polygons = os.path.join(DATA, "polygons.shp")
    raster = os.path.join(DATA, "slope_nodata.tif")
    with rasterio.open(raster) as src:
        arr = src.read(1)
        affine = src.transform

    # without nodata
    stats = zonal_stats(polygons, arr, affine=affine, stats=["nodata", "count", "min"])
    assert stats[0]["min"] == -9999.0
    assert stats[0]["nodata"] == 0
    assert stats[0]["count"] == 75

    # with nodata
    stats = zonal_stats(
        polygons, arr, affine=affine, nodata=-9999.0, stats=["nodata", "count", "min"]
    )
    assert stats[0]["min"] >= 0.0
    assert stats[0]["nodata"] == 36
    assert stats[0]["count"] == 39


def test_transform():
    with rasterio.open(raster) as src:
        arr = src.read(1)
        affine = src.transform
    polygons = os.path.join(DATA, "polygons.shp")

    stats = zonal_stats(polygons, arr, affine=affine)
    with pytest.deprecated_call():
        stats2 = zonal_stats(polygons, arr, transform=affine.to_gdal())
    assert stats == stats2


def test_prefix():
    polygons = os.path.join(DATA, "polygons.shp")
    stats = zonal_stats(polygons, raster, prefix="TEST")
    for key in ["count", "min", "max", "mean"]:
        assert key not in stats[0]
    for key in ["TESTcount", "TESTmin", "TESTmax", "TESTmean"]:
        assert key in stats[0]


def test_geojson_out():
    polygons = os.path.join(DATA, "polygons.shp")
    features = zonal_stats(polygons, raster, geojson_out=True)
    for feature in features:
        assert feature["type"] == "Feature"
        assert "id" in feature["properties"]  # from orig
        assert "count" in feature["properties"]  # from zonal stats


# do not think this is actually testing the line i wanted it to
# since the read_features func for this data type is generating
# the properties field
def test_geojson_out_with_no_properties():
    polygon = Polygon([[0, 0], [0, 0.5], [1, 1.5], [1.5, 2], [2, 2], [2, 0]])
    arr = np.array([[100, 1], [100, 1]])
    affine = Affine(1, 0, 0, 0, -1, 2)

    stats = zonal_stats(polygon, arr, affine=affine, geojson_out=True)
    assert "properties" in stats[0]
    for key in ["count", "min", "max", "mean"]:
        assert key in stats[0]["properties"]

    assert stats[0]["properties"]["mean"] == 34


# remove when copy_properties alias is removed
def test_copy_properties_warn():
    polygons = os.path.join(DATA, "polygons.shp")
    # run once to trigger any other unrelated deprecation warnings
    # so the test does not catch them instead
    stats_a = zonal_stats(polygons, raster)
    with pytest.deprecated_call():
        stats_b = zonal_stats(polygons, raster, copy_properties=True)
    assert stats_a == stats_b


def test_nan_counts():
    from affine import Affine

    transform = Affine(1, 0, 1, 0, -1, 3)

    data = np.array([[np.nan, np.nan, np.nan], [0, 0, 0], [1, 4, 5]])

    # geom extends an additional row to left
    geom = "POLYGON ((1 0, 4 0, 4 3, 1 3, 1 0))"

    # nan stat is requested
    stats = zonal_stats(geom, data, affine=transform, nodata=0.0, stats="*")

    for res in stats:
        assert res["count"] == 3  # 3 pixels of valid data
        assert res["nodata"] == 3  # 3 pixels of nodata
        assert res["nan"] == 3  # 3 pixels of nans

    # nan are ignored if nan stat is not requested
    stats = zonal_stats(geom, data, affine=transform, nodata=0.0, stats="count nodata")

    for res in stats:
        assert res["count"] == 3  # 3 pixels of valid data
        assert res["nodata"] == 3  # 3 pixels of nodata
        assert "nan" not in res


# Optional tests
def test_geodataframe_zonal():
    gpd = pytest.importorskip("geopandas")

    polygons = os.path.join(DATA, "polygons.shp")
    df = gpd.read_file(polygons)
    if not hasattr(df, "__geo_interface__"):
        pytest.skip("This version of geopandas doesn't support df.__geo_interface__")

    expected = zonal_stats(polygons, raster)
    assert zonal_stats(df, raster) == expected


#%% # add a unit test for the function gdf_zonal_stats

def gdf_zonal_stats(vectors, *args, **kwargs):
    """The primary zonal statistics entry point.
    All arguments are passed directly to ``gen_zonal_stats``.
    See its docstring for details.
    
    The difference is that ``gdf_zonal_stats`` will
    return a GeoDataFrame rather than a generator.
    Besides this, we make sure the new gdf as the right
    metadata by conserving the CRS
    """

    gdf = gpd.GeoDataFrame.from_features(
        gen_zonal_stats(vectors, geojson_out=True, *args, **kwargs)
        )
    
    # if the input is a file path : open has gdf and get crs
    if isinstance(vectors, str): 
        gdf.crs = gpd.read_file(vectors).crs

    # if the vectors has a 'crs' attribute use it to keep the crs
    elif hasattr(vectors, "crs"):
        gdf.crs = vectors.crs
        
    # otherwise, we cannot store the crs ? (not sure if that is possible)
    else:
        gdf.crs = None
        print("Warning : the crs is not stored in the output GeoDataFrame")

    return gdf


def test_gdf_zonal_stats():
    gpd = pytest.importorskip("geopandas")
    polygons = os.path.join(DATA, "polygons.shp")
    gdf = gpd.read_file(polygons)
    if not hasattr(gdf, "__geo_interface__"):
        pytest.skip("This version of geopandas doesn't support df.__geo_interface__")

    expected_with_path = gdf_zonal_stats(polygons, raster)
    expected_with_gdf = gdf_zonal_stats(gdf, raster)
    assert expected_with_gdf.equals(expected_with_path) # Use equals() to compare DataFrames
    assert expected_with_gdf.crs == gdf.crs             # Check that the crs is conserved


# TODO #  gen_zonal_stats(<features_without_props>)
# TODO #  gen_zonal_stats(stats=nodata)
# TODO #  gen_zonal_stats(<raster with non-integer dtype>)
# TODO #  gen_zonal_stats(transform AND affine>)
