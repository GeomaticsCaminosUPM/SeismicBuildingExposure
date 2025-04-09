import rasterio as rio
import scipy
import numpy as np
import pandas as pd
import geopandas as gpd 
import osmnx as ox
from pyproj import CRS
import warnings
import shapely

def get_shape(file):
    """
    Returns the shape (height, width) of a raster file.
    
    :param file: Path to the raster file or an open rasterio dataset.
    :return: (height, width) tuple.
    """
    if isinstance(file, str):  # If file is a path, open it
        with rio.open(file) as src:
            return src.width, src.height
    else:  # If file is an already open rasterio dataset
        return file.width, file.height

def validate_crs(src:rio.io.DatasetReader|rio.io.DatasetWriter|dict|CRS|str|int):
    import re
    if type(src) is dict:
        crs = src['crs']
    elif type(src) is CRS:
        crs = src
    elif type(src) is str:
        crs = CRS.from_string(src)
    elif type(src) is int:
        crs = CRS.from_epsg(src)
    else:
        try:
            crs = CRS(src.crs) 
        except:
            raise Exception(f"src type {type(src)} not accepted: {src}")

    warnings.filterwarnings(
        "ignore",
        message=re.escape(
            "You will likely lose important projection information when converting to a PROJ string from another format. See: https://proj.org/faq.html#what-is-the-best-format-for-describing-coordinate-reference-systems"
        ),
        category=UserWarning
    )
    if len(crs.to_proj4()) == 0:
        crs_str = crs.to_wkt()
        
        if "LOCAL_CS" in crs_str:
            if "ETRS89-extended / LAEA Europe" in crs_str:
                crs = CRS.from_epsg(3035)

                if (type(src) == rio.io.DatasetReader) and (src.mode != 'r'):
                    src.crs = crs

                return crs
            # Add more mappings as needed
            # elif "Another projection" in crs_str:
            #     return CRS.from_epsg(some_epsg_code)
            else:
                raise ValueError("Unknown LOCAL_CS definition; manual intervention needed.")
        else:
            raise ValueError("CRS is invalid, but not due to LOCAL_CS.")
    else:
        return crs.to_epsg() # to_proj4()

def driver_and_extension(driver):
    if driver == 'GTiff' or driver == "tif" or driver == ".tif":
        extension = ".tif"
        driver = 'GTiff'
    elif driver == 'JPEG' or driver == "jpg" or driver == ".jpg": 
        extension = ".jpg"
        driver = 'JPEG'
    elif driver == 'PNG' or driver == "png" or driver == ".png": 
        extension = ".png"
        driver = 'PNG'
    else:
        raise Exception(f"driver {driver} not implemented")
    
    return driver, extension

def get_crs(file):
    if type(file) is str:
        try:
            src = rio.open(file,'r+')
        except:
            src = rio.open(file,'r')
    else:
        src = file

    crs = validate_crs(src)
    if type(file) is str:    
        src.close()
    
    return crs

def bounds(file,crs=None):
    from pyproj import Transformer
    from pyproj import CRS
    
    if type(file) is str:
        try:
            src = rio.open(file,'r+')
        except:
            src = rio.open(file,'r')

        file_crs = CRS.from_epsg(get_crs(src))
        src.close()
    else:
        src = file
        file_crs = get_crs(src)

    r = gpd.GeoSeries(shapely.geometry.box(src.bounds.left,src.bounds.bottom,src.bounds.right,src.bounds.top),crs=file_crs)
    if crs is not None: 
        r = r.to_crs(crs)

    return r

def resolution_to_shape(file,resolution:tuple):
    if type(file) == str:
        img_bounds = bounds(file)
    else: 
        img_bounds = file.copy()
        
    img_bounds = img_bounds.to_crs(img_bounds.estimate_utm_crs())
    shape = (
        int(np.ceil((img_bounds.total_bounds[2] - img_bounds.total_bounds[0]) / resolution[0])),
        int(np.ceil((img_bounds.total_bounds[3] - img_bounds.total_bounds[1]) / resolution[1])),
    )
    return shape

def read(file:str,geometry,shape:tuple=None,resolution:tuple=None,nodata=np.nan,function=None,all_touched:bool=False,crop:bool=True):
    with rio.open(file) as src:
        if geometry is not None:
            geometry = geometry.geometry.copy()
            geometry = geometry.to_crs(src.crs)  # Ensure roads are in the same CRS as raster
        
        if resolution is not None: 
            shape = resolution_to_shape(file,resolution)

        if shape is not None:
            resampled_data = src.read(
                out_shape=(src.count, shape[1], shape[0]),  # (bands, height, width)
                resampling=rio.enums.Resampling.bilinear
            )

            if geometry is None:
                return resampled_data

            # Compute new transform after resampling
            new_transform = src.transform * src.transform.scale(
                (src.width / resampled_data.shape[2]),  # Adjust width scale
                (src.height / resampled_data.shape[1])  # Adjust height scale
            )
            
            # Create an in-memory raster with the same metadata as the source raster
            with rio.io.MemoryFile() as memfile:
                # Write the resampled data to the in-memory raster
                with memfile.open(
                    driver='GTiff',
                    count=src.count,
                    dtype=resampled_data.dtype,
                    crs=src.crs,
                    transform=new_transform,
                    width=resampled_data.shape[2],
                    height=resampled_data.shape[1]
                ) as memsrc:
                    # Write the data to the in-memory file
                    memsrc.write(resampled_data)
                    def read_geometry(geom):
                        if geom.is_empty == True:
                            return None 

                        try:
                            values,_ = rio.mask.mask(
                                memsrc,  # Use the in-memory raster
                                shapes=[geom],  # Vector geometries to mask
                                crop=crop,  # Keep original raster size
                                nodata = nodata,
                                all_touched=all_touched
                            )
                        except Exception as e:
                            if "Input shapes do not overlap raster" not in str(e):
                                raise Exception(e)
                             
                            values = None

                        if function is not None:
                            values = function(values) 
                            if type(values) == tuple: 
                                values = pd.Series(values)

                        return values

                    return geometry.apply(read_geometry)
        else:
            if geometry is None:
                return rio.read(src)

            def read_geometry(geom):
                if geom.is_empty == True:
                    return None 
                    
                try:
                    values, _ = rio.mask.mask(
                        src,  # Use the in-memory raster
                        shapes=[geom],  # Vector geometries to mask
                        crop = crop,  # Keep original raster size
                        nodata = nodata,
                        all_touched=all_touched
                    )
                except Exception as e:
                    print(f"Error processing polygon: {e}")
                    values = None

                if function is not None:
                    values = function(values) 
                    if type(values) == tuple: 
                        values = pd.Series(values)

                return values

            return geometry.apply(read_geometry)


def save(output_path:str, arr, bounds:gpd.GeoSeries, driver:str = "PNG"):
    # driver available here https://gdal.org/drivers/raster/index.html
    driver, extension = driver_and_extension(driver)
    output_path = output_path.split('.')[0] + extension

    # Extract the GeoSeries bounds
    crs = bounds.crs
    img_bounds = gpd.GeoSeries(shapely.geometry.box(*bounds.total_bounds),crs=crs)
    utm = img_bounds.estimate_utm_crs()
    if abs(bounds.to_crs(utm).union_all().area - img_bounds.to_crs(utm).union_all().area) > 10**-4:
        warnings.warn(
            "The rasterized image bounds do not match exactly with the provided bounds. It covers a larger area.",
            UserWarning
        )
        
    minx,miny,maxx,maxy = img_bounds.total_bounds

    if len(arr.shape) == 2:
        arr = arr[np.newaxis,:,:]
        
    if len(arr.shape) == 3:
        n_bands = arr.shape[0]
        y_shape = arr.shape[1]
        x_shape = arr.shape[2]
    else:
        raise Exception(f"shape of array is {arr.shape} but is should have the shape (n_bands,x_shape,y_shape)")

    if n_bands > 3: 
        raise Exception("The array should have the shape (n_bands,x_shape,y_shape)")
        
    # Calculate the pixel size
    x_pixel_size = (maxx - minx) / x_shape
    y_pixel_size = (maxy - miny) / y_shape

    # Create the transformation
    transform = rio.transform.from_bounds(minx, miny, maxx, maxy, x_shape, y_shape)

    dtype = np.min_scalar_type(np.max(arr))
    if 'int64' in str(dtype):
        dtype = np.float64

    if 'uint16' in str(dtype):
        dtype = np.uint32

    if 'int16' in str(dtype):
        dtype = np.int32

    if 'float16' in str(dtype):
        dtype = np.float32

    # Create the GeoTIFF file
    with rio.open(
        output_path,
        'w',
        driver=driver,
        height=y_shape,
        width=x_shape,
        count=n_bands,  # Number of bands (RGB)
        dtype=dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(arr)



def download_osm_streets(bounds:gpd.GeoSeries,network_type:str='drive',custom_filter=None):
    if bounds.crs.is_projected == True:
        crs = bounds.crs 
    else:
        crs = bounds.estimate_utm_crs()
    #network_type (string {"all", "all_public", "bike", "drive", "drive_service", "walk"}) – what type of street network to get if custom_filter is None
    G=ox.graph.graph_from_polygon(bounds.to_crs(4326).union_all(), network_type=network_type, simplify=True, retain_all=False, truncate_by_edge=True,custom_filter=custom_filter)
    G=ox.projection.project_graph(G,to_crs=crs)
    edges = ox.graph_to_gdfs(G,nodes=False)

    return edges

def sample_points(geoms, interval):
    """Sample points every `interval` meters along a Shapely MultiLineString."""
    mls = geoms.union_all()
    total_length = mls.length
    sampled_points = []
    
    # Generate distances at fixed intervals
    distances = np.arange(0, total_length, interval)
    
    # Extract points at these distances
    for dist in distances:
        sampled_points.append(mls.interpolate(dist))
    
    return gpd.GeoSeries(sampled_points,crs=geoms.crs)


def points_gdf_edge_gradient(gdf,value_column='height', k=5,min_distance=10):
    from scipy.spatial import cKDTree
    if k+1 > len(gdf): 
        k = len(gdf) - 1 

    df_orig_index = pd.DataFrame({'index':gdf.index})
    gdf_copy = gdf[[value_column,'geometry']].copy()
    gdf_copy['index'] = gdf_copy.index
    gdf_copy = gdf_copy.drop_duplicates(['geometry',value_column]).reset_index(drop=True)
    # Extract coordinates and heights
    coords = np.array(list(zip(gdf_copy.geometry.x, gdf_copy.geometry.y)))
    heights = gdf_copy[value_column].values
    
    # Build a KDTree for fast nearest neighbor search
    tree = cKDTree(coords)
    
    # Query the nearest `k+1` points (including self)
    distances, indices = tree.query(coords, k=k+1)


    near_indices = (distances <= min_distance) * indices
    near_heights = heights[near_indices] * (distances <= min_distance)
    near_heights[distances > min_distance] = np.inf

    delete_ids = np.argwhere(np.min(near_heights,axis=1) != near_heights[:,0])
    delete_ids = delete_ids.transpose()[0]

    if len(delete_ids) > 0:
        gdf_copy = gdf_copy.drop(index=delete_ids)
        if k+1 > len(gdf_copy): 
            k = len(gdf_copy) - 1 

        # Extract coordinates and heights
        coords = np.array(list(zip(gdf_copy.geometry.x, gdf_copy.geometry.y)))
        heights = gdf_copy[value_column].values
        # Build a KDTree for fast nearest neighbor search
        tree = cKDTree(coords)
        # Query the nearest `k+1` points (including self)
        distances, indices = tree.query(coords, k=k+1)
        
    # Compute height differences
    height_diffs = heights[:, None] - heights[indices]

    distances[distances == 0] = 10**-20
    # Compute gradients (ignoring divide-by-zero concerns)
    gradients = height_diffs / distances

    # Ignore self-comparison by setting the first column to -inf
    abs_gradients = np.abs(gradients)
    abs_gradients[:, 0] = -np.inf
    gradients[:, 0] = -np.inf

    # Get row indices (0,1,2,...,num_points) and corresponding max gradient indices
    row_indices = np.arange(gradients.shape[0])
    max_grad_indices = np.argmax(abs_gradients, axis=1)

    # Use advanced indexing to get the max gradient for each point
    gdf_copy['gradient'] = gradients[row_indices, max_grad_indices]
    df = df_orig_index.merge(gdf_copy[['index','gradient']],on='index',how='left')
    return list(df['gradient'])

def create_TIN_raster(points_gdf,img_bounds,shape:tuple=None,resolution:tuple=None,value_column='height'):
    from rasterio.transform import from_bounds
    from scipy.spatial import Delaunay
    from scipy.interpolate import LinearNDInterpolator

    points_gdf = points_gdf.to_crs(img_bounds.crs).copy()
    if resolution is not None:
        shape = resolution_to_shape(img_bounds,resolution)

    data = pd.DataFrame({'x':points_gdf.geometry.x,'y':points_gdf.geometry.y,'values':points_gdf[value_column]})
    # Extract coordinates and values
    points = np.column_stack((data["x"], data["y"]))
    values = data["values"].values

    # Perform Delaunay triangulation
    tri = Delaunay(points)

    # Extract given bounds and shape
    xmin, ymin, xmax, ymax = img_bounds.total_bounds
    width, height = shape

    # Create grid based on the given shape and bounds
    grid_x, grid_y = np.meshgrid(
        np.linspace(xmax, xmin, width),
        np.linspace(ymin, ymax, height)
    )
    interpolator = LinearNDInterpolator(tri, values)
    grid_z = interpolator(grid_x,grid_y)

    return grid_z



def _cast(collection):
    """
    Cast a collection to a shapely geometry array.
    """
    try:
        import  geopandas as gpd
        import shapely
    except (ImportError, ModuleNotFoundError) as exception:
        raise type(exception)(
            "shapely and gpd are required for shape statistics."
        ) from None

    if Version(shapely.__version__) < Version("2"):
        raise ImportError("Shapely 2.0 or newer is required.")

    if isinstance(collection, gpd.GeoSeries | gpd.GeoDataFrame):
        return np.asarray(collection.geometry.array)
    else:
        if isinstance(collection, np.ndarray | list):
            return np.asarray(collection)
        else:
            return np.array([collection])

        

def _ring_inertia_x_y(polygon, reference_point):
    """
    Using equation listed on en.wikipedia.org/wiki/Second_moment_of_area#Any_polygon, the second
    moment of area is the sum of the inertia across the x and y axes:

    The :math:`x` axis is given by:

    .. math::

        I_x = (1/12)\\sum^{N}_{i=1} (x_i y_{i+1} - x_{i+1}y_i) (x_i^2 + x_ix_{i+1} + x_{i+1}^2)

    While the :math:`y` axis is in a similar form:

    .. math::

        I_y = (1/12)\\sum^{N}_{i=1} (x_i y_{i+1} - x_{i+1}y_i) (y_i^2 + y_iy_{i+1} + y_{i+1}^2)

    where :math:`x_i`, :math:`y_i` is the current point and :math:`x_{i+1}`, :math:`y_{i+1}` is the next point,
    and where :math:`x_{n+1} = x_1, y_{n+1} = y_1`. For multipart polygons with holes,
    all parts are treated as separate contributions to the overall centroid, which
    provides the same result as if all parts with holes are separately computed, and then
    merged together using the parallel axis theorem.

    References
    ----------
    Hally, D. 1987. The calculations of the moments of polygons. Canadian National
    Defense Research and Development Technical Memorandum 87/209.
    https://apps.dtic.mil/dtic/tr/fulltext/u2/a183444.pdf

    """

    coordinates = shapely.get_coordinates(polygon)
    centroid = shapely.centroid(polygon)
    centroid_coords = shapely.get_coordinates(centroid)
    points = coordinates - centroid_coords

    # Ensure reference_point is a Shapely Point
    if not isinstance(reference_point, Point):
        reference_point = Point(reference_point)  # Convert to Point if necessary

    I_x = np.abs(np.sum(
        (points[:-1, 0]**2 + points[:-1, 0] * points[1:, 0] + points[1:, 0]**2) *
        (points[1:, 1] * points[:-1, 0] - points[:-1, 1] * points[1:, 0])
    ) / 12)

    I_y = np.abs(np.sum(
        (points[:-1, 1]**2 + points[:-1, 1] * points[1:, 1] + points[1:, 1]**2) *
        (points[1:, 1] * points[:-1, 0] - points[:-1, 1] * points[1:, 0])
    ) / 12)

    I_xy = np.abs(np.sum(
        (points[:-1,0] * points[1:,1] + 2 * points[:-1,0] * points[:-1,1] + 2 * points[1:,0] * points[1:,1] + points[1:,0] * points[:-1,1]) *
        (points[1:, 1] * points[:-1, 0] - points[:-1, 1] * points[1:, 0])
    ) / 24)
    
    # Step 4: Use the Parallel Axis Theorem to shift the moments of inertia to the new reference point

    d_x = abs(reference_point.x - polygon.centroid.x)  # Distance along the x-axis
    d_y = abs(reference_point.y - polygon.centroid.y)  # Distance along the y-axis

    A = polygon.area  # Area of the polygon
    I_x += A * d_x ** 2
    I_y += A * d_y ** 2
    I_xy += A * d_x * d_y

    return I_x, I_y, I_xy

def calc_inertia_all(collection):
    """
    Calculate inertia in x and y dirs.
    """

    # Ensure the collection is in the right format for computation
    ga = _cast(collection)  # Assuming _cast is a helper function to ensure compatibility with geopandas

    # Get the fundamental parts of the collection
    parts, collection_ix = shapely.get_parts(ga, return_index=True)
    rings, ring_ix = shapely.get_rings(parts, return_index=True)

    # Get the exterior and interior rings
    collection_ix = np.repeat(
        collection_ix, shapely.get_num_interior_rings(parts) + 1
    )

    polygon_rings = shapely.polygons(rings)
    is_external = np.zeros_like(collection_ix).astype(bool)
    is_external[0] = True
    is_external[1:] = ring_ix[1:] != ring_ix[:-1]

    # Create GeoDataFrame to work with the polygons
    polygon_rings = gpd.GeoDataFrame(
        dict(
            collection_ix=collection_ix,
            ring_within_geom_ix=ring_ix,
            is_external_ring=is_external,
        ),
        geometry=polygon_rings,
    )

    polygon_rings["sign"] = (1 - polygon_rings.is_external_ring * 2) * -1

    # Get the original centroids for each polygon
    original_centroids = shapely.centroid(ga)
    polygon_rings["collection_centroid"] = original_centroids[collection_ix]
    
    # Apply the principal moment calculation for all polygons at once
    polygon_rings[["I_x", "I_y", "I_xy"]] = polygon_rings.apply(
        lambda x: pd.Series(_ring_inertia_x_y(x['geometry'], x['collection_centroid'])) * x['sign'],
        axis=1
    )

    # Aggregate the moments for each collection
    aggregated_inertia = polygon_rings.groupby("collection_ix")[["I_x", "I_y", "I_xy"]].sum()
    return aggregated_inertia['I_x'], aggregated_inertia['I_y'], aggregated_inertia['I_xy']

def calc_inertia_principal(collection,principal_dirs:bool=False):
    """
    Calculate the principal moments of inertia for a collection of polygons.
    """
    I_x, I_y, I_xy = calc_inertia_all(collection)
    aggregated_inertia = pd.DataFrame({'I_x':I_x,'I_y':I_y,'I_xy':I_xy})
    
    aggregated_inertia['I_tensor'] = aggregated_inertia.apply(
        lambda row: np.array([[row['I_x'], - row['I_xy']],
                             [- row['I_xy'], row['I_y']]]), axis=1
    )

    # Calculate the eigenvalues (principal moments of inertia) and eigenvectors (principal axes)
    if principal_dirs:
        result = aggregated_inertia['I_tensor'].apply(lambda tensor: pd.Series(np.linalg.eig(tensor)))
        result = result.apply(
            lambda x: pd.Series([x[0][1],x[0][0],x[1][1],x[1][0]])
            if float(x[0][0]) < float(x[0][1]) 
            else pd.Series([x[0][0],x[0][1],x[1][0]*np.array([1,-1]),x[1][1]*np.array([1,-1])]),
            axis=1
        )

        vect_1 = np.stack(result[2])
        vect_2 = np.stack(result[3])
        printcipal_mom_1 = result[0]
        printcipal_mom_2 = result[1]

        return np.array(printcipal_mom_1), np.array(vect_1), np.array(printcipal_mom_2), np.array(vect_2)
    else:
        result = aggregated_inertia['I_tensor'].apply(lambda tensor: pd.Series(np.sort(np.linalg.eigvals(tensor))))
        printcipal_mom_1 = result[1]
        printcipal_mom_2 = result[0]
        return np.array(printcipal_mom_1), np.array(printcipal_mom_2)
        
def inertia_slenderness(geoms:gpd.GeoDataFrame) -> list:
    """
    Calculates the inertia irregularity index for building footprint polygons comparing the principal components of the inertia tensor (max and min).

    Parameters:
        geoms (gpd.GeoDataFrame): A GeoDataFrame containing building footprint geometries as polygons.

    Returns:
        list: A list with the same order as geoms which contains the calculated Polsby-Popper index for each geometry.
    """
    geoms = geoms.copy() 
    # Ensure the geometries are in a projected CRS for accurate area and length calculations
    if not geoms.crs.is_projected:
        geoms = geoms.to_crs(geoms.geometry.estimate_utm_crs())

    I_max, I_min = calc_inertia_principal(geoms.geometry,principal_dirs=False)
    return list(np.sqrt(I_min / I_max))

def circunsribed_slenderness(geoms:gpd.GeoDataFrame) -> list:
    geoms = geoms.copy() 
    # Ensure the geometries are in a projected CRS for accurate area and length calculations
    if not geoms.crs.is_projected:
        geoms = geoms.to_crs(geoms.geometry.estimate_utm_crs())

    inertia_df = calc_inertia_principal(geoms, principal_dirs=True)

    total_length_1, total_length_2 = circunscribed_square(
        geoms.geometry,
        inertia_df[1][:,0],
        inertia_df[1][:,1],
        inertia_df[3][:,0],
        inertia_df[3][:,1],
        return_length=True
    )

    return list(np.maximum(np.array(total_length_1) / np.array(total_length_2),
                  np.array(total_length_2) / np.array(total_length_1)))  
