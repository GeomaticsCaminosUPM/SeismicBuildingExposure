# helper funcs

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
