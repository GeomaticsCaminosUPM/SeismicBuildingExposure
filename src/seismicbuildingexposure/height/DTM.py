from .utils import read, bounds, download_osm_streets, get_crs, sample_points, points_gdf_edge_gradient, create_TIN_raster
from scipy.ndimage import gaussian_filter

def from_DSM(
  dsm_path,
  resolution,
  mask:gpd.GeoSeries=None,
  sampling_distance=100,
  nodata=-1000,
  max_gradient=1,
  smooth:bool=True,
  img_bounds:gpd.GeoSeries=None
):
    from scipy.ndimage import gaussian_filter

    multiply = 4

    if img_bounds is None:
        img_bounds = bounds(dsm_path)
    else:
        img_bounds = img_bounds.to_crs(get_crs(dsm_path))
        img_bounds = gpd.GeoSeries(shapely.box(img_bounds.total_bounds))

    roads = download_osm_streets(bounds(dsm_path))
    roads = roads[roads.intersects(img_bounds.to_crs(roads.crs).union_all())]
    if mask is not None:
        mask = mask[mask.intersects(img_bounds.to_crs(mask.crs).union_all())]

    roads = roads.to_crs(roads.estimate_utm_crs())
    points = gpd.GeoDataFrame({},geometry=sample_points(roads,max(resolution)*sampling_distance))

    points_buffer = points.geometry.to_crs(points.estimate_utm_crs()).buffer(max(resolution)*sampling_distance*multiply) 
    points_buffer = points_buffer.intersection(roads.to_crs(points_buffer.crs).buffer(5*max(resolution)).union_all())
    if mask is not None:
        points_buffer = points_buffer.difference(mask.to_crs(points_buffer.crs).buffer(5*max(resolution)).union_all())
    
    points = points[points_buffer.is_empty == False]
    points_buffer = points_buffer[points_buffer.is_empty == False]

    def read_dtm_points(pixels):
        if pixels is None:
            return None 

        pixels = np.array(pixels[0,:,:]).flatten()
        pixels = pixels[np.isnan(pixels) == False]
        pixels = pixels[pixels != nodata]

        if len(pixels) < 0.25*(sampling_distance*multiply*5):
            return None 

        # Create the histogram (frequency distribution)
        hist, bin_edges = np.histogram(pixels, bins=5)
        hist = hist / np.sum(hist)
        m = np.mean(hist) + np.std(hist)
        hist = np.array([0,*hist])

        # Find the peaks in the histogram
        peak_ids, _ = scipy.signal.find_peaks(hist,height=m)
        if len(peak_ids) == 0:
            if (bin_edges[-1] - bin_edges[0]) < (max(resolution) * 10):
                return np.median(pixels)
            else:
                return None
                    
        p = np.min(peak_ids) - 1 
        value = np.median(pixels[(pixels > bin_edges[p]) & (pixels < bin_edges[p+1])])

        return value

    points['height'] = read(dsm_path,points_buffer,all_touched=True,function=read_dtm_points,crop=True,resolution=resolution,nodata=nodata)
    points['gradient'] = points_gdf_edge_gradient(points,value_column='height',k=50,min_distance=max(resolution)*sampling_distance)
    points = points.loc[points['gradient'].isna() == False]
    points = points.loc[points['gradient'] < max_gradient*np.min(resolution)]
    points['gradient'] = points_gdf_edge_gradient(points,value_column='height',k=10,min_distance=max(resolution)*sampling_distance)
    points = points.loc[points['gradient'].isna() == False]
    points = points.loc[points['gradient'] < max_gradient*np.min(resolution)]
    points = points.loc[points['gradient'] > -max_gradient*np.min(resolution)]

    dtm = create_TIN_raster(points,img_bounds=img_bounds,resolution=resolution,value_column='height')

    gradient = np.ma.masked_where(np.isnan(dtm),dtm)
    gradient = np.sqrt(np.gradient(gradient,axis=0)**2+np.gradient(gradient,axis=1)**2).filled(nodata)
    
    a = img_bounds.total_bounds[2] - img_bounds.total_bounds[0]
    b = img_bounds.total_bounds[3] - img_bounds.total_bounds[1]
    points_x = np.round(((np.array(points.to_crs(img_bounds.crs).geometry.x) - img_bounds.total_bounds[0]) / a)*(gradient.shape[1]-1)).astype(int) 
    points_y = np.round(((np.array(points.to_crs(img_bounds.crs).geometry.y) - img_bounds.total_bounds[1]) / b)*(gradient.shape[0]-1)).astype(int)

    points['gradient'] = 0 
    for i in [-1,0,1]:
        for j in [-1,0,1]:
            x_idx = points_x + i  
            y_idx = points_y + j   
            x_idx = np.clip(x_idx,0,gradient.shape[1]-1) 
            y_idx = np.clip(y_idx,0,gradient.shape[0]-1) 
            gradient_i = gradient[y_idx,x_idx] 
            points['gradient'] = np.maximum(points['gradient'],gradient_i)
    
    orig_len = len(points)
    points = points.loc[points['gradient'].isna() == False]
    points = points.loc[points['gradient'] >= 0]
    points = points.loc[points['gradient'] < max_gradient]

    if len(points) < orig_len:
        dtm = create_TIN_raster(points,img_bounds=img_bounds,resolution=resolution,value_column='height')       

    if smooth:
        dtm = gaussian_filter(dtm, sigma=sampling_distance, radius=sampling_distance)

    dtm[np.isnan(dtm)] = nodata

    return dtm
