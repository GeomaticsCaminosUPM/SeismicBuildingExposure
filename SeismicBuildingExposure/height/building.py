def get_height_and_steepness(dsm_path:str,resolution:tuple,footprints:gpd.GeoSeries,nodata=-1000,maximum_mode:bool=True):
    gdf = gpd.GeoDataFrame({'id':footprints.index},geometry=footprints.geometry,crs=footprints.crs)
    orig_gdf = gdf.copy()
    gdf = gdf.to_crs(gdf.estimate_utm_crs())
    gdf['pixels']=read(dsm_path,geometry=gdf,resolution=resolution,crop=True,nodata=nodata)
    gdf = gdf[gdf['pixels'].isna() == False]
    gdf['pixels'] = gdf['pixels'].apply(lambda x : x[0,:,:])
    gdf['pixels'] = gdf['pixels'].apply(lambda x : None if np.all(x == nodata) else x)
    gdf = gdf[gdf['pixels'].isna() == False]


    def get_height(pixels):
        valid_pixels = pixels[pixels != nodata]
        if len(valid_pixels) == 0:
            return None 
            
        hist, bin_edges = np.histogram(valid_pixels,bins=5)
        hist = hist / np.sum(hist)
        m = np.mean(hist) + np.std(hist)
        if maximum_mode:
            hist = np.array([*hist,-1])
        else:
            hist = np.array([-1,*hist])

        # Find the peaks in the histogram
        peak_ids, _ = scipy.signal.find_peaks(hist,height=m)
        if len(peak_ids) == 0:
            if (bin_edges[-1] - bin_edges[0]) < (max(resolution) * 10):
                return np.median(valid_pixels)
            else:
                return None

         
        p = np.max(peak_ids) 
        if maximum_mode == False:
            p -= 1

        value = np.median(valid_pixels[(valid_pixels > bin_edges[p]) & (valid_pixels < bin_edges[p+1])])

        return value

    gdf['height'] = gdf['pixels'].apply(get_height)

    def get_gradient(pixels):
        masked_array = np.ma.masked_where(pixels==nodata,pixels)
        gradient = np.sqrt((np.gradient(masked_array,axis=0) / resolution[0])**2 + (np.gradient(masked_array,axis=1) / resolution[1])**2)
        gradient = gradient.compressed()
        m = np.mean(gradient)
        s = np.std(gradient)

        return np.mean(gradient[gradient < (m+s)])

    gdf['mean_steepness'] = gdf['pixels'].apply(get_gradient)

    gdf = orig_gdf.merge(gdf[['id','height','mean_steepness']],on='id',how='left')

    return pd.DataFrame({'height':list(gdf['height']), 'mean_steepness':list(gdf['mean_steepness'])})

def roof_data(dsm_path:str,resolution:tuple,footprints:gpd.GeoSeries,nodata=-1000):
    return get_height_and_steepness(dsm_path=dsm_path,resolution=resolution,footprints=footprints,nodata=nodata,maximum_mode=True)

def ground_data(dtm_path:str,resolution:tuple,footprints:gpd.GeoSeries,nodata=-1000):
    return get_height_and_steepness(dsm_path=dtm_path,resolution=resolution,footprints=footprints,nodata=nodata,maximum_mode=False)

def data(dsm_path:str,dtm_path:str,resolution:tuple,footprints:gpd.GeoSeries,nodata=-1000):
  footprints = footprints.copy()
  footprints['building_height'] = result_dsm['height'] - result_dtm['height']
  footprints['ground_altitude'] = result_dtm['height']
  footprints['ground_steepness'] = result_dtm['mean_steepness']
  return footprints[['building_height','ground_altitude','ground_steepness']]


def roof_parts(dsm_path:str,resolution:tuple,footprints:gpd.GeoSeries,nodata=-1000):
    gdf = gpd.GeoDataFrame({'id':footprints.index},geometry=footprints.geometry,crs=footprints.crs)
    orig_gdf = gdf.copy()
    gdf = gdf.to_crs(gdf.estimate_utm_crs())
    gdf['pixels']=read(dsm_path,geometry=gdf,resolution=resolution,crop=True,nodata=nodata)
    gdf = gdf[gdf['pixels'].isna() == False]
    gdf['pixels'] = gdf['pixels'].apply(lambda x : x[0,:,:])
    gdf['pixels'] = gdf['pixels'].apply(lambda x : None if np.all(x == nodata) else x)
    gdf = gdf[gdf['pixels'].isna() == False]
    src_crs = get_crs(dsm_path)
    gdf = gdf.to_crs(src_crs)

    def part_polygon(pixels,footprint):
        import scipy
        valid_pixels = pixels[pixels != nodata]
        if len(valid_pixels) == 0:
            return None, None 
            
        hist, bin_edges = np.histogram(valid_pixels,bins=20)
        hist = hist / np.sum(hist)
        peak_ids, _ = scipy.signal.find_peaks(np.append(hist,[0]),height=0.01)

        height, width = pixels.shape
        transform = rio.transform.from_bounds(*footprint.bounds, width, height)
        heights = bin_edges[peak_ids-1]
        raster_heights = np.zeros(pixels.shape,dtype=np.int32)

        for h in heights:
            raster_heights[pixels > h] = np.int32(h*10**4)

        raster_heights = rio.features.sieve(raster_heights,int(round(4/(np.max(resolution)**2))))

        mask = raster_heights != 0
        # Vectorize the raster
        geometries = (
            {'properties': {'height': v / 10**4}, 'geometry': s}
            for i, (s, v) in enumerate(
                rio.features.shapes(raster_heights, mask=mask, transform=transform)
            )
        )

        # Convert to GeoDataFrame
        geometries = list(geometries) 
        geometries = gpd.GeoDataFrame.from_features(geometries)
        geometries = geometries[geometries.isna() == False]
        if len(geometries) == 0:
            return None, None

        return list(geometries['geometry']), list(geometries['height'])

    gdf[['parts','part_heights']] = gdf.apply(lambda x : pd.Series(part_polygon(x['pixels'],x['geometry'])),axis=1)
    gdf_parts = gdf[['id','parts']].explode('parts').reset_index(drop=True)

    def get_gradient(values):
        if values is None:
            return None 

        if len(values) == 0:
            return None 
            
        values = values[0,:,:]
        masked_array = np.ma.masked_where(values==nodata,values)
        gradient = np.sqrt((np.gradient(masked_array,axis=0) / resolution[0])**2 + (np.gradient(masked_array,axis=1) / resolution[1])**2)
        gradient = gradient.compressed()
        m = np.mean(gradient)
        s = np.std(gradient)

        return np.mean(gradient[gradient < (m+s)])

    gdf_parts = gdf_parts.loc[gdf_parts['parts'].isna() == False]

    gdf_parts['part_steepness'] = read(dsm_path,geometry=gpd.GeoSeries(gdf_parts['parts'],crs=gdf.crs),resolution=resolution,crop=True,nodata=nodata,function=get_gradient)
    gdf_parts = gdf_parts.groupby('id').agg(list).reset_index()

    gdf = orig_gdf.merge(gdf[['id','parts','part_heights']],on='id',how='left')
    gdf = gdf.merge(gdf_parts[['id','part_steepness']],on='id',how='left')

    return pd.DataFrame({'parts':gdf['parts'],'part_height':gdf['part_heights'],'part_steepness':gdf['part_steepness']})


