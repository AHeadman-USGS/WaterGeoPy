import shapefile, pycrs
from pyproj import Proj, Transformer
from shapely.geometry import shape


class Shp:
    def __init__(self, path):
        self.path = path
        self.prj = self.path[:-4] + ".prj"
        self.shp = shapefile.Reader(self.path)
        self.feature = self.shp.shapeRecords()[0]
        self.geo_int = self.feature.__geo_interface__
        self.shapely = shape(self.geo_int['geometry'])
        self.x_cen, self.y_cen = self._centroid()
        self.daymet_x, self.daymet_y = self.daymet_proj()

    def _centroid(self):
        center = self.shapely.centroid.coords
        center_x = center[0][0]
        center_y = center[0][1]
        return center_x, center_y

    def daymet_proj(self):
        scrs = pycrs.load.from_file(self.prj)
        daymet_proj = pycrs.load.from_file("shapefiles//Daymet.prj")
        transformer = Transformer.from_crs(crs.to_proj4(), daymet_proj.to_proj4())
        return transformer.transform(self.x_cen, self.y_cen)


check = Shp(path="shapefiles//RobinsonForest.shp")