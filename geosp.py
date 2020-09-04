from osgeo import ogr, gdal, osr
import numpy as np
import json
import pycrs



class Shp:
    def __init__(self, path):
        self.path = path
        self.shp = ogr.Open(self.path)
        self.prj = self.shp.GetLayer().GetSpatialRef().ExportToProj4()
        self.feature = self.shp.GetLayer(0).GetFeature(0)
        self.extent = feature.GetGeometryRef().GetEnvelope()
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
        transformer = pycrs.Transformer.from_crs(self.prj, daymet_proj.to_proj4())
        return transformer.transform(self.x_cen, self.y_cen)


raster_path = "shapefiles//HA00_AWC.tif"
vector_path = "shapefiles//RobinsonForest.shp"



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
    return (x1, y1, xsize, ysize)





def zonal_stats(raster_path, vector_path):

    r_data = gdal.Open(raster_path)
    r_band = r_data.GetRasterBand(1)
    r_geotransform = r_data.GetGeoTransform()
    v_data = ogr.Open(vector_path)
    v_feature = v_data.GetLayer(0)

    sourceprj = v_feature.GetSpatialRef()
    targetprj = osr.SpatialReference(wkt=r_data.GetProjection())
    to_fill = ogr.GetDriverByName("Esri Shapefile")
    ds = to_fill.CreateDataSource("shapefiles//projected.shp")
    outlayer = ds.CreateLayer('', targetprj, ogr.wkbPolygon)
    outlayer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))

    for feature in v_feature:
        transform = osr.CoordinateTransformation(sourceprj, targetprj)
        transformed = feature.GetGeometryRef()
        transformed.Transform(transform)


        geom = ogr.CreateGeometryFromWkb(transformed.ExportToWkb())
        defn = outlayer.GetLayerDefn()
        feat = ogr.Feature(defn)
        feat.SetField('id', 0)
        feat.SetGeometry(geom)
        outlayer.CreateFeature(feat)
        feat = None

    ds = None

    v_data = ogr.Open("shapefiles//projected.shp")
    v_feature = v_data.GetLayer(0)


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
        mask = np.logical_not(v_to_r_array)

    )

    feature_stats = {
        'min': float(masked.min()),
        'mean': float(masked.mean()),
        'max': float(masked.max()),
        'std': float(masked.std()),
        'sum': float(masked.sum()),
        'count': int(masked.count())
    }

    stats.append(feature_stats)









check = Shp(path="shapefiles//RobinsonForest.shp")
print(check.daymet_x)
print(check.daymet_y)



file = ogr.Open("shapefiles//RobinsonForest.shp")  ##
feature = file.GetLayer(0).GetFeature(0)
feature_json = feature.ExportToJson()
extent = feature.GetGeometryRef().GetEnvelope()
centoid = json.loads( ##
    feature.GetGeometryRef().Centroid().ExportToJson()) ##

cente_x = centroid['coordinates'][0] ##
cente_y = centroid['coordinates'][1] ##
prj = file.GetLayer().GetSpatialRef().ExportToProj4()  ##
daymet_proj = pycrs.load.from_file("shapefiles//Daymet.prj") ##
transformer = Transformer.from_crs(prj, daymet_proj.to_proj4())
test = transformer.transform(center_x, center_y)
print(test)

