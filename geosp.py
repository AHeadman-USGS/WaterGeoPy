from osgeo import ogr, gdal, osr, gdalconst
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
        self.lyr = self.shp.GetLayer()
        self.prj = self.shp.GetLayer().GetSpatialRef()
        self.prj4 = self.prj.ExportToProj4()
        self.feature = self.shp.GetLayer(0).GetFeature(0)
        self.extent = self.feature.GetGeometryRef().GetEnvelope()
        self.x_cen, self.y_cen = self._centroid()
        self.daymet_x, self.daymet_y = self.daymet_proj()
        self.karst_flag = 0

    @classmethod
    def _clean(cls, path):
        ds = ogr.Open(path, 1)
        lyr = ds.GetLayer()
        defn = lyr.GetLayerDefn()
        for i in range(defn.GetFieldCount()):
            name = defn.GetFieldDefn(i).GetName()
            if name == "Shape_Area" or name == "Shape_Leng":
                lyr.DeleteField(i)
            else:
                continue
        ds.Destroy()
        clean_shp = Shp(path=path)
        return clean_shp

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


class dbShp:
    """
    Basically the same as Raster class, for shapes provided with DB or created by code such as _karst and _karstclip
    shps.
    """
    def __init__(self, path):
        self.path = path
        self.shp = ogr.Open(self.path)
        self.lyr = self.shp.GetLayer()
        self.prj = self.shp.GetLayer().GetSpatialRef()
        self.prj4 = self.prj.ExportToProj4()
        self.feature = None  # needs an assignment from outside the class.  Fine.
        # self.extent = self.feature.GetGeometryRef().GetEnvelope()
        # self.x_cen, self.y_cen = self._centroid()
        # self.daymet_x, self.daymet_y = self.daymet_proj()

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

def karst_detection(raster, shp):
    """
    :param raster: Raster class object built from karst raster.
    :param shp: SHP class object from entire basin.
    :return: Shp.karst_flag will be triggered, or it won't.
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

    stats = []  # Keeping this unless there are several features in the same shapefile.

    driver = gdal.GetDriverByName('MEM')
    v_to_r = driver.Create('', src_offset[2], src_offset[3], 1, gdal.GDT_Byte)
    v_to_r.SetGeoTransform(new_gt)
    gdal.RasterizeLayer(v_to_r, [1], v_feature, burn_values=[1])
    v_to_r_array = v_to_r.ReadAsArray()
    masked = np.ma.MaskedArray(
        src_array,
        mask=np.logical_not(v_to_r_array)

    )

    if masked.max() > 0:
        return 1
    else:
        return 0





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
    return feature_stats


def twi_bins(raster, shp, nbins=30):

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

    v_to_r = driver.Create('', src_offset[2], src_offset[3], 1, gdal.GDT_Byte)
    v_to_r.SetGeoTransform(new_gt)
    gdal.RasterizeLayer(v_to_r, [1], v_feature, burn_values=[1])
    v_to_r_array = v_to_r.ReadAsArray()
    masked = np.ma.MaskedArray(
        src_array,
        mask=np.logical_or(
            np.logical_not(v_to_r_array),
            src_array < 0)

    )

    mx = masked.max()
    mn = masked.min()
    mean = masked.mean()
    intvl = (mx - mn) / (nbins + 1)
    edges = np.arange(mn, mx, intvl)
    histo = np.histogram(masked, bins=edges)


    # need mean of each bin.  Get the rest of the stats while there.
    # TWI Mean is the value we need for TopModel Input.

    bins = []

    for i in range(nbins):
        line = []
        bin = i + 1
        twi_val = histo[1][i]
        proportion = histo[0][i]/np.sum(histo[0])

        line.append(bin)
        line.append(twi_val)
        line.append(proportion)
        bins.append(line)

    return bins


def clip(src, shp):
    """
    :param src: shapefile class with karst polygons.  sinks.shp in db.
    :param shp: shapefile class with basin boundary.
    :return: shapefile output and class with karst.
    """
    driver = ogr.GetDriverByName("ESRI Shapefile")
    src_ds = driver.Open(src.path, 0)
    src_layer = src_ds.GetLayer()

    clip_ds = driver.Open(shp.path, 0)
    clip_layer = clip_ds.GetLayer()
    srs = osr.SpatialReference()
    srs.ImportFromProj4(src.prj4)

    out_path = shp.path[:-4] + '_karst_.shp'

    if os.path.exists(out_path):
        driver.DeleteDataSource(out_path)

    out_ds = driver.CreateDataSource(out_path)
    out_layer = out_ds.CreateLayer('', srs=srs, geom_type=ogr.wkbMultiPolygon)

    ogr.Layer.Clip(src_layer, clip_layer, out_layer)

    karstshp = dbShp(path=out_path)
    return karstshp


def erase(src, diff):
    driver = ogr.GetDriverByName("ESRI Shapefile")
    src_ds = driver.Open(src.path, 0)
    src_layer = src_ds.GetLayer()
    src_feature = src_layer.GetFeature(0)

    diff_ds = driver.Open(diff.path, 0)
    diff_layer = diff_ds.GetLayer()
    diff_feature = diff_layer.GetFeature(0)
    srs = osr.SpatialReference()
    srs.ImportFromProj4(src.prj4)

    out_path = shp.path[:-4] + '_notkarst.shp'
    if os.path.exists(out_path):
        driver.DeleteDataSource(out_path)

    out_ds = driver.CreateDataSource(out_path)
    out_layer = out_ds.CreateLayer('', srs=srs, geom_type=ogr.wkbMultiPolygon)
    out_defn = out_layer.GetLayerDefn()
    out_feature = ogr.Feature(out_defn)
    src_geom = src_feature.GetGeometryRef()
    diff_geom = diff_feature.GetGeometryRef()
    src_diff = src_geom.Difference(diff_geom)
    out_feature.SetGeometry(src_diff)

    wkt = out_feature.geometry().ExportToWkt()
    out_layer.CreateFeature(out_feature)
    karstless = dbShp(path=out_path)
    return karstless

def dissolve_polygon(raster, shp):
    """
    Need to work on size management here.

    :param raster: use karst_raster or any soil raster.  We just need the GT object
    :param shp: shp file to be dissolved
    :return: raster object of dissolved shp file.
    """
    gt = raster.gt()
    x_min = gt[0]
    y_max = gt[3]
    x_res = raster.data.RasterXSize
    y_res = raster.data.RasterYSize
    x_max = x_min + gt[1] * x_res
    y_min = y_max + gt[5] * y_res
    pixel_width = gt[1]
    out_file = "shapefiles//karst_flat.shp"
    target_ds = gdal.GetDriverByName('MEM').Create('', x_res, y_res, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform((x_min, pixel_width, 0, y_min, 0, pixel_width))
    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(0)
    band.FlushCache()
    gdal.RasterizeLayer(target_ds, [1], shp.lyr, options=["ATTRIBUTE=Gridcode"])
    driver = ogr.GetDriverByName("ESRI Shapefile")
    out_ds = driver.CreateDataSource(out_file)
    srs = osr.SpatialReference()
    srs.ImportFromProj4(shp.prj4)
    out_lyr = out_ds.CreateLayer(out_file, srs=srs)
    fd = ogr.FieldDefn("DN", ogr.OFTInteger)
    out_lyr.CreateField(fd)
    gdal.Polygonize(band, band, out_lyr, -1, [])
    multi = ogr.Geometry(ogr.wkbMultiPolygon)
    for feat in out_lyr:
        if feat.geometry():
            feat.geometry().CloseRings()  # this copies the first point to the end
            wkt = feat.geometry().ExportToWkt()
            multi.AddGeometryDirectly(ogr.CreateGeometryFromWkt(wkt))
            out_lyr.DeleteFeature(feat.GetFID())
    union = multi.UnionCascaded()
    out_feat = ogr.Feature(out_lyr.GetLayerDefn())
    out_feat.SetGeometry(union)
    out_lyr.CreateFeature(out_feat)

    flat = dbShp(path=out_file)
    target_ds = None
    return flat






zone_stats = []
shp = Shp._clean(path="shapefiles//RockCastle.shp")
karst_raster = Raster(path="database//Sinks.tif")
karst_shp = Shp(path="shapefiles/sinks2.shp")

shp.karst_flag = karst_detection(karst_raster, shp)

if shp.karst_flag == 1:
    karst = clip(karst_shp, shp)
    karst_new = Shp(path=karst.path)
    karst_flat = dissolve_polygon(karst_raster, karst_new)
    karst_f = Shp(path=karst_flat.path)
    karst_sub = erase(shp, karst_f)
   # basin_no_karst = Shp(path=karst_sub.path)

for file in os.listdir("database"):
    if file.startswith("HA00") and file.endswith(".tif"):
        raster = Raster(path="database//{}".format(file))
        zonal = zonal_stats(raster, shp)
        zone_stats.append(zonal)

    elif file.startswith("TWI.tif") and file.endswith(".tif"):
        raster = Raster(path="database//{}".format(file))
        twi = twi_bins(raster, shp)



print("--- %s seconds ---" % (time.time() - start_time))
print(zone_stats)
print(twi)
print(shp.karst_flag)