def setback_ratio(
    footprints,
    min_setback_height=1,
    min_symmetry_distance=1,
    height_limit=0.15,
    height_factor=3,
    symmetry_factor=0.5/0.3,
    ):

    footprints = footprints.copy()
    footprints['id'] = footprints.index
    orig_df = footprints[['id']].copy()

    footprints = footprints[footprints['parts'].isna() == False]
    footprints = footprints[footprints['ground_altitude'].isna() == False]
    footprints = footprints[footprints['building_height'].isna() == False]
    footprints = footprints[footprints['part_height'].isna() == False]
    floor_heights = footprints['ground_altitude']
    building_heights = footprints['building_height']
    roof_parts = footprints['parts']
    roof_part_heights = footprints['part_height']
    result_df = footprints[['id']].copy()
    footprints = footprints.geometry


    setback_ratio_a = _setback_ratio(
        footprints=footprints,
        floor_heights=floor_heights,
        building_heights=building_heights,
        roof_parts=roof_parts,
        roof_part_heights=roof_part_heights,
        min_setback_height=min_setback_height,
        min_symmetry_distance=min_symmetry_distance,
        height_limit=height_limit,
        height_factor=height_factor,
        symmetry_factor=symmetry_factor
    ) 

    setback_ratio_b = _setback_ratio(
        footprints=footprints,
        floor_heights=floor_heights,
        building_heights=building_heights,
        roof_parts=roof_parts,
        roof_part_heights=roof_part_heights,
        min_setback_height=min_setback_height,
        min_symmetry_distance=min_symmetry_distance,
        height_limit=height_limit,
        height_factor=height_factor,
        symmetry_factor=symmetry_factor,
        rotate=True
    ) 

    result = np.maximum(setback_ratio_a, setback_ratio_b)
    result_df['setback_ratio'] = list(result)
    orig_df = orig_df.merge(result_df[['id','setback_ratio']],on='id',how='left')
    return list(orig_df['setback_ratio'])

def _setback_ratio(
    footprints,
    floor_heights,
    building_heights,
    roof_parts,
    roof_part_heights,
    min_setback_height=1,
    min_symmetry_distance=1,
    height_limit=0.15,
    height_factor=3,
    symmetry_factor=0.5/0.3,
    rotate:bool=False
    ):

    from shapely.geometry import LineString
    gdf = pd.DataFrame({
        'floor_heights':floor_heights,
        'building_heights':building_heights,
        'roof_parts':roof_parts,
        'roof_part_heights':roof_part_heights,
        'geometry':footprints
    })

    orig_crs = footprints.crs
    utm_crs = footprints.estimate_utm_crs()
    gdf['roof_parts'] = gdf['roof_parts'].apply(lambda x:list(gpd.GeoSeries(x,crs=orig_crs).to_crs(utm_crs)))
    gdf['roof_parts_copy'] = gdf['roof_parts'].copy()
    gdf['roof_part_heights_copy'] = gdf['roof_part_heights'].copy()
    gdf['index'] = gdf.index

    orig_df = pd.DataFrame({'index':gdf['index']})

    gdf = gdf.explode(['roof_parts','roof_part_heights']).reset_index(drop=True)
    gdf = gpd.GeoDataFrame(gdf,geometry='geometry',crs=orig_crs).to_crs(utm_crs)
    gdf['roof_parts'] = gpd.GeoSeries(gdf['roof_parts'],crs=gdf.crs)
    gdf['nearest_point'] = gdf.geometry.boundary.interpolate(gdf.geometry.boundary.project(gdf['roof_parts'].centroid))

    gdf['distance'] = np.sqrt((
        gdf.geometry.bounds['maxx']-gdf.geometry.bounds['minx']
    )**2 + (
        gdf.geometry.bounds['maxy']-gdf.geometry.bounds['miny']
    )**2) / 2

    gdf['vect_length'] = np.sqrt((gdf['roof_parts'].centroid.x - gdf['nearest_point'].x) ** 2 + (gdf['roof_parts'].centroid.y - gdf['nearest_point'].y) ** 2)

    if rotate:
        gdf['line_start_y'] = gdf['roof_parts'].centroid.y - (-(gdf['roof_parts'].centroid.x - gdf['nearest_point'].x) / gdf['vect_length']) * (gdf['distance'] + 1) 
        gdf['line_start_x'] = gdf['roof_parts'].centroid.x - ((gdf['roof_parts'].centroid.y - gdf['nearest_point'].y) / gdf['vect_length']) * (gdf['distance'] + 1) 
        gdf['line_end_y'] = gdf['roof_parts'].centroid.y + (-(gdf['roof_parts'].centroid.x - gdf['nearest_point'].x) / gdf['vect_length']) * (gdf['distance'] + 1) 
        gdf['line_end_x'] = gdf['roof_parts'].centroid.x + ((gdf['roof_parts'].centroid.y - gdf['nearest_point'].y) / gdf['vect_length']) * (gdf['distance'] + 1)  
    else:
        gdf['line_start_x'] = gdf['roof_parts'].centroid.x - ((gdf['roof_parts'].centroid.x - gdf['nearest_point'].x) / gdf['vect_length']) * (gdf['distance'] + 1) 
        gdf['line_start_y'] = gdf['roof_parts'].centroid.y - ((gdf['roof_parts'].centroid.y - gdf['nearest_point'].y) / gdf['vect_length']) * (gdf['distance'] + 1) 
        gdf['line_end_x'] = gdf['roof_parts'].centroid.x + ((gdf['roof_parts'].centroid.x - gdf['nearest_point'].x) / gdf['vect_length']) * (gdf['distance'] + 1) 
        gdf['line_end_y'] = gdf['roof_parts'].centroid.y + ((gdf['roof_parts'].centroid.y - gdf['nearest_point'].y) / gdf['vect_length']) * (gdf['distance'] + 1)   

    gdf['line'] = gpd.GeoSeries(gdf.apply(lambda row: LineString([(row['line_start_x'],row['line_start_y']),(row['line_end_x'],row['line_end_y'])]),axis=1),crs=gdf.crs)

    gdf['cross_section'] = gdf.geometry.intersection(gdf['line'])

    gdf = gdf.explode(column='cross_section').reset_index(drop=True)
    gdf['L'] = gdf['cross_section'].length

    gdf = gdf.loc[gdf['cross_section'].distance(gdf['roof_parts'].centroid) < 10**-3]
    gdf['height_meassure_points'] = gdf['roof_parts'].buffer(0.25).boundary.intersection(gdf['cross_section'])

    gdf['height_bool'] = gdf.apply(lambda x:shapely.intersects(x['roof_parts_copy'],x['height_meassure_points']),axis=1)
    gdf['H1'] = gdf.apply(lambda x: np.array(x['roof_part_heights_copy'])[np.array(x['height_bool'])],axis=1)
    gdf['H1'] = gdf.apply(lambda x: x['H1'][x['H1'] < x['roof_part_heights']],axis=1)
    gdf['H1'] = gdf['H1'].apply(lambda x: None if len(x) == 0 else np.max(x))
    gdf = gdf.loc[gdf['H1'].isna()==False]
    gdf['H1'] = np.abs(gdf['roof_part_heights'] - gdf['H1'])
    gdf = gdf.drop(columns=['roof_parts_copy','height_meassure_points','height_bool'])

    gdf['width_line'] = gdf['cross_section'].difference(gdf['roof_parts'])
    gdf['L1'] = gdf['width_line'].length
    gdf['setback_symmetry'] = shapely.distance(gdf['width_line'].centroid,gdf['roof_parts'].centroid) < min_symmetry_distance

    gdf['setback_ratio'] = gdf['L1'] / gdf['L']
    gdf = gdf.loc[(gdf['H1']/gdf['L1']) > min_setback_height]
    gdf.loc[gdf['setback_symmetry'],'setback_ratio'] *= symmetry_factor 
    gdf.loc[gdf['building_heights']*height_limit+gdf['floor_heights'] > gdf['roof_part_heights'],'setback_ratio'] *= height_factor 
    gdf = gdf.groupby('index').agg({'setback_ratio': 'max'}).reset_index()
    
    orig_df = orig_df.merge(gdf[['index','setback_ratio']],how='left',on='index')
    orig_df.loc[orig_df['setback_ratio'].isna()] = 0

    return list(orig_df['setback_ratio'])

def inertia_slenderness(geoms:gpd.GeodataFrame,height_column:str='building_height'):
    if geoms.crs.is_projected == False:
        geoms = geoms.to_crs(geoms.geometry.estimate_utm_crs())
        
    slenderness = utils.inertia_slenderness(geoms)
    return geoms[height_column] / np.sqrt(geoms.area * (1/slenderness))
