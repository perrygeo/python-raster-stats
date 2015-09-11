# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
import click
from rasterstats import zonal_stats
from rasterstats.utils import combine_features_results
from rasterstats.io import read_features
import logging
try:
    import simplejson as json
except:
    import json

SETTINGS = dict(help_option_names=['-h', '--help'])
version = 0.1


@click.command(context_settings=SETTINGS)
@click.argument('input-geojson', type=click.File('r'), default='-')
@click.argument('output-geojson', type=click.File('w'), default='-')
@click.version_option(version=version, message='%(version)s')
@click.option('--raster', '-r', required=True, type=click.Path(exists=True))
@click.option('--all-touched/--no-all-touched', default=False)
@click.option('--band', type=int, default=1)
@click.option('--categorical/--no-categorical', default=False)
@click.option('--global-src-extent/--no-global-src-extent', default=False)
@click.option('--indent', type=int, default=None)
@click.option('--info/--no-info', default=False)
@click.option('--nodata', type=int, default=None)
@click.option('--prefix', type=str, default='_')
@click.option('--stats', type=str, default=None)
def zonalstats(input_geojson, raster, output_geojson, all_touched, band, categorical,
               global_src_extent, indent, info, nodata, prefix, stats):
    '''zonalstats generates summary statistics of geospatial raster datasets
    based on vector features.

    The input and output arguments of zonalstats should be valid GeoJSON FeatureCollections. The output GeoJSON will be mostly unchanged but have additional properties per feature describing the summary statistics (min, max, mean, etc.) of the underlying raster dataset. The input and output arguments default to stdin and stdout but can also be file paths.

    The raster is specified by the required -r/--raster argument.

    Example, calculate rainfall stats for each state and output to file:

    \b
    zonalstats states.geojson -r rainfall.tif > mean_rainfall_by_state.geojson
    '''

    if info:
        logging.basicConfig(level=logging.INFO)
    mapping = json.loads(input_geojson.read())
    input_geojson.close()
    try:
        if mapping['type'] == "FeatureCollection":
            feature_collection = mapping
        else:
            feature_collection = {'type': 'FeatureCollection'}
        features = read_features(mapping)
    except (AssertionError, KeyError):
        raise ValueError("input_geojson must be valid GeoJSON")

    if stats is not None:
        stats = stats.split(" ")
        if 'all' in [x.lower() for x in stats]:
            stats = "ALL"

    zonal_results = zonal_stats(
        features,
        raster,
        all_touched=all_touched,
        band_num=band,
        categorical=categorical,
        global_src_extent=global_src_extent,
        nodata_value=nodata,
        stats=stats,
        copy_properties=False)

    feature_collection['features'] = list(
        combine_features_results(features, zonal_results, prefix))

    output_geojson.write(json.dumps(feature_collection, indent=indent))
    output_geojson.write("\n")
