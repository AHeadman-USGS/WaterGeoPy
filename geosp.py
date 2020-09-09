from osgeo import ogr, gdal, osr
import os
import numpy as np
import json
import pycrs
import pyproj


class Shp:
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
    def __init__(self, path):
        self.path = path
        self.data = gdal.Open(self.path)
        self.band_1 = self.data.GetRasterBand(1)
        self.gt = self.data.GetGeoTransform
        self.prj = osr.SpatialReference(wkt=self.data.GetProjection())
        self.prj4 = self.prj.ExportToProj4()


# raster_path = "shapefiles//HA00_AWC.tif"
# vector_path = "shapefiles//RobinsonForest.shp"
#
# raster = Raster(path = "shapefiles//HA00_AWC.tif")
# shp = Shp(path="shapefiles//RobinsonForest.shp")

def bbox_to_pixel_offsets(gt, bbox):
    originX = gt[0]
    originY = gt[3]
    pixel_width = gt[1]
    pixel_height = gt[5]
    x1 = int((bbox[0] - originX) / pixel_width)
    x2 = int((bbox[1] - originX) / pixel_width) + 1

    y1 = int((bbox[3] - originY) / pixel_height)
    y2 = int((bbox[2] - originY) / pixel_height) + 1

    xsize = x2 - x1
    ysize = y2 - y1
    return x1, y1, xsize, ysize


def zonal_stats(raster, shp):

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


        # for feature in v_feature:
        #     transform = osr.CoordinateTransformation(sourceprj, targetprj)
        #     transformed = feature.GetGeometryRef()
        #     transformed.Transform(transform)
        #     geom = ogr.CreateGeometryFromWkb(transformed.ExportToWkb())
        #     defn = outlayer.GetLayerDefn()
        #     feat = ogr.Feature(defn)
        #     #feat.SetField('id', 0)
        #     feat.SetGeometry(geom)
        #     outlayer.CreateFeature(feat.Clone())
        #     feat = None
        # ds = None

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


raster = Raster(path = "shapefiles//HA00_AWC.tif")
shp = Shp(path="shapefiles//RobinsonForest.shp")

# stats = []
# for file in os.listdir("database"):
#     if file.endswith(".tif"):
#         raster = Raster(path="database//{}".format(file))
#         zonal = zonal_stats(raster, shp)
#         stats.append(zonal)

test = zonal_stats(raster, shp)
print(test)
