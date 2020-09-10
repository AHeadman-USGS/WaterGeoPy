

from osgeo import ogr, gdal, osr
import os
import numpy as np
import json
import pycrs
import pyproj

import time
start_time = time.time()

class Shp:
    """
    Contains various fuctions and metadata desc in init related in SHP objects.
    While titled SHP, currently this should only be used with polygons.  Will incorporate fun things like
    points in future versions.  I currently see no reason to incorporate lines.
    Outside reliance on the daymet.prj (included in ./static/geospatial) to transform things into daymet to build the
    temp/precip series.
    """

    def __init__(self, path):
        self.path = path
        self.shp = ogr.Open(self.path)
        self.prj = self.shp.GetLayer().GetSpatialRef()
        self.prj4 = self.prj.ExportToProj4()
        self.feature = self.shp.GetLayer(0).GetFeature(0)
        self.extent = self.feature.GetGeometryRef().GetEnvelope()
        self.x_cen, self.y_cen = self._centroid()
        self.daymet_x, self.daymet_y = self.daymet_proj()

    def _centroid(self):
        centroid = json.loads(
            self.feature.GetGeometryRef().Centroid().ExportToJson())
        center_x = centroid['coordinates'][0]
        center_y = centroid['coordinates'][1]
        return center_x, center_y

    def daymet_proj(self):
        daymet_proj = pycrs.load.from_file("shapefiles//Daymet.prj")
        transformer = pyproj.Transformer.from_crs(self.prj4, daymet_proj.to_proj4())
        return transformer.transform(self.x_cen, self.y_cen)


class Raster:
    """
       Contains various fuctions and metadata desc in init related in rasters objects.
       WaterPy internals (./static/geospatal/rasters) utilizes tifs, this object class is compatible with any
       osgeo/gdal compliant raster formats.
    """

    def __init__(self, path):
        self.path = path
        self.data = gdal.Open(self.path)
        self.band_1 = self.data.GetRasterBand(1)
        self.gt = self.data.GetGeoTransform
        self.prj = osr.SpatialReference(wkt=self.data.GetProjection())
        self.prj4 = self.prj.ExportToProj4()


def bbox_to_pixel_offsets(gt, bbox):
    """
    Function to offset (aka snap) polygon to raster.


    :param gt: geotransform variable from gdal.data.GetGeoTransform
    :param bbox: Bounding extent coordinates from ogr.feature.GetExtent()
    :return: tuple to use as a multiplier for the raster array.
    """

    origin_x = gt[0]
    origin_y = gt[3]
    pixel_width = gt[1]
    pixel_height = gt[5]
    x1 = int((bbox[0] - origin_x) / pixel_width)
    x2 = int((bbox[1] - origin_x) / pixel_width) + 1

    y1 = int((bbox[3] - origin_y) / pixel_height)
    y2 = int((bbox[2] - origin_y) / pixel_height) + 1

    xsize = x2 - x1
    ysize = y2 - y1
    return x1, y1, xsize, ysize


def zonal_stats(raster, shp):
    """
    Converts a shp file into a raster mask.  Masks off a polygon and extracts statistics from the area within the mask.
    Currently this only works with a shp file with one feature, however, it's written so that it could be adjusted to
    handle multiple features.

    :param raster: Raster class object.
    :param shp: Shp class object.
    :return: list of dict objects from computed stats.
    """

    r_data = raster.data
    r_band = r_data.GetRasterBand(1)
    r_geotransform = raster.gt()
    v_data = shp.shp
    v_feature = v_data.GetLayer(0)

    sourceprj = v_feature.GetSpatialRef()
    targetprj = osr.SpatialReference(wkt=r_data.GetProjection())

    if sourceprj.ExportToProj4() != targetprj.ExportToProj4():
        to_fill = ogr.GetDriverByName('Memory')
        ds = to_fill.CreateDataSource("project")
        outlayer = ds.CreateLayer('poly', targetprj, ogr.wkbPolygon)
        feature = v_feature.GetFeature(0)
        transform = osr.CoordinateTransformation(sourceprj, targetprj)
        transformed = feature.GetGeometryRef()
        transformed.Transform(transform)
        geom = ogr.CreateGeometryFromWkb(transformed.ExportToWkb())
        defn = outlayer.GetLayerDefn()
        feat = ogr.Feature(defn)
        feat.SetGeometry(geom)
        outlayer.CreateFeature(feat.Clone())
        feat = None
        v_feature = outlayer

    src_offset = bbox_to_pixel_offsets(r_geotransform, v_feature.GetExtent())
    src_array = r_band.ReadAsArray(*src_offset)

    new_gt = (
        (r_geotransform[0] + (src_offset[0] * r_geotransform[1])),
        r_geotransform[1], 0.0,
        (r_geotransform[3] + (src_offset[1] * r_geotransform[5])),
        0.0, r_geotransform[5]
    )

    driver = gdal.GetDriverByName('MEM')

    stats = []

    v_to_r = driver.Create('', src_offset[2], src_offset[3], 1, gdal.GDT_Byte)
    v_to_r.SetGeoTransform(new_gt)
    gdal.RasterizeLayer(v_to_r, [1], v_feature, burn_values=[1])
    v_to_r_array = v_to_r.ReadAsArray()
    masked = np.ma.MaskedArray(
        src_array,
        mask=np.logical_not(v_to_r_array)

    )

    feature_stats = {
        'source': str(raster.path),
        'min': float(masked.min()),
        'mean': float(masked.mean()),
        'max': float(masked.max()),
        'std': float(masked.std())
    }

    ds = None

    stats.append(feature_stats)
    return stats


shp = Shp(path="shapefiles//GrapeVine.shp")

stats = []
for file in os.listdir("database"):
    if file.startswith("HA00") and file.endswith(".tif"):
        raster = Raster(path="database//{}".format(file))
        zonal = zonal_stats(raster, shp)
        stats.append(zonal)

print(stats)
print("--- %s seconds ---" % (time.time() - start_time))