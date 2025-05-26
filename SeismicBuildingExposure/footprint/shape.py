import geopandas as gpd 
import pandas as pd
import shapely 
from shapely.geometry import Polygon, MultiPolygon, LineString, Point
import numpy as np
import warnings
from .utils import (
    get_normal,
    explode_edges,
    explode_exterior_and_interior_rings,
    calc_inertia_z,
    eq_circle_intertia,
    calc_inertia_all,
    calc_inertia_principal,
    get_angle,
    center_of_mass,
    min_bbox,
    rectangle_to_directions,
    circunscribed_setback_length,
    circunscribed_rectangle
)


def convex_hull_momentum(geoms:gpd.GeoDataFrame|gpd.GeoSeries) -> list:
    """TODO: Explore normalize by the boundary and not by the hull and some type of normalization 0-1. Explore polsby + shape"""
    """
    Calculates an index to quantify the irregularity of building footprints.

    The irregularity index is computed using the formula:
        Irregularity = (l * d) / L
    where:
        - `l`: Length of the segments outside the convex hull of the shape.
        - `d`: Distance from the center of gravity of the segments outside the hull to the hull.
        - `L`: Total perimeter length of the convex hull.

    Parameters:
        geoms (gpd.GeoDataFrame): A GeoDataFrame containing building footprint geometries as polygons.

    Returns:
        list: A list with the same order as geoms which contains the computed irregularity index for each geometry.
    """
    geoms_copy = geoms.copy()
    if type(geoms) is gpd.GeoSeries:
        geoms = gpd.GeoDataFrame({},geometry=geoms,crs=geoms.crs)
        
    geoms_copy['geom_id'] = geoms_copy.index.copy()

    if not geoms_copy.crs.is_projected:
        geoms_copy = geoms_copy.to_crs(geoms_copy.geometry.estimate_utm_crs())


    geoms_copy = explode_exterior_and_interior_rings(geoms_copy)
    boundary = geoms_copy.geometry.copy()
    convex_hull = geoms_copy.geometry.convex_hull.boundary.copy()
    geoms_copy['hull_length'] = convex_hull.length
    geoms_copy['boundary_length'] = boundary.length

    geoms_copy.geometry = shapely.difference(boundary, convex_hull.buffer(0.005)) #Path through irregularity
    geoms_copy['hull_geom'] = shapely.difference(convex_hull, boundary.buffer(0.005)) #Path through hull
    geoms_copy = geoms_copy.loc[geoms_copy.geometry.is_empty == False]

    geoms_copy = geoms_copy.explode(index_parts=False).reset_index(drop=True)

    geoms_copy = explode_edges(geoms_copy,min_length=0)

    geoms_copy[['edge_center','normal_vector']] = geoms_copy.apply(lambda x: pd.Series(get_normal(x['edges'],1)),axis=1)

    geoms_copy['distance_to_hull'] = geoms_copy.apply(lambda x: shapely.distance(x['edge_center'],x['hull_geom']),axis=1)
    geoms_copy['edge_length'] = geoms_copy['edges'].length

    geoms_copy['shape_irregularity'] = (geoms_copy['distance_to_hull'] * geoms_copy['edge_length']) / geoms_copy['hull_length']

    #geoms_copy.loc[0:len(geoms_copy)-1,'normal_1'] = geoms_copy.loc[1:len(geoms_copy),'normal'].reset_index(drop=True)
    #geoms_copy.loc[0:len(geoms_copy)-1,'geom_id_1'] = geoms_copy.loc[1:len(geoms_copy),'geom_id'].reset_index(drop=True)
    #geoms_copy['angle_irregularity'] = geoms_copy.apply(lambda x: pd.Series(get_angle_sharp(x['normal'],x['normal_1'],x['geom_id'],x['geom_id_1'])),axis=1)

    geoms_copy = geoms_copy.groupby('geom_id').agg({'shape_irregularity':'sum'})

    result = geoms.merge(geoms_copy[['shape_irregularity']],right_index=True,left_index=True,how='left')
    result.loc[result['shape_irregularity'].isna(),'shape_irregularity'] = 0 
    result['shape_irregularity'] = result['shape_irregularity'].astype(float)

    return list(result['shape_irregularity'])

def polsby_popper(geoms:gpd.GeoDataFrame|gpd.GeoSeries, fill_holes:bool=True) -> list:
    """TODO: Polsby popper donut shape. boundary.length takes both inner and outer circle into accout so that the perimeter is very large"""
    """
    Calculates the Polsby-Popper index for building footprint polygons.

    The Polsby-Popper index is a measure of shape compactness, defined by the formula:
        Polsby-Popper Index = (4 * π * A) / P²
    where:
        - `A`: The area of the polygon.
        - `P`: The perimeter of the polygon.

    Parameters:
        geoms (gpd.GeoDataFrame): A GeoDataFrame containing building footprint geometries as polygons.
        fill_holes (bool, optional): Fill all holes of the geometries (interior patios, etc.) to compute the index. Defaults to True.

    Returns:
        list: A list in the same order as geoms which contains the calculated Polsby-Popper index for each geometry.
    """
    geoms = geoms.copy()
    if type(geoms) is gpd.GeoSeries:
        geoms = gpd.GeoDataFrame({},geometry=geoms,crs=geoms.crs)
    # Ensure the geometries are in a projected CRS for accurate area and length calculations
    if not geoms.crs.is_projected:
        geoms = geoms.to_crs(geoms.geometry.estimate_utm_crs())
    
    # Calculate the Polsby-Popper compactness score
    if fill_holes:
        geoms_holes_filled = geoms.geometry.apply(
            lambda x: Polygon(x.exterior)
        )
        geoms['polsby_popper'] = (4 * np.pi * geoms_holes_filled.geometry.area) / (geoms_holes_filled.geometry.boundary.length ** 2)
    else:
        geoms['polsby_popper'] = (4 * np.pi * geoms.geometry.area) / (geoms.geometry.boundary.length ** 2)
    
    return list(geoms['polsby_popper']) 
    
def inertia_slenderness(geoms:gpd.GeoDataFrame|gpd.GeoSeries) -> list:
    """
    Calculates the inertia irregularity index for building footprint polygons comparing the principal components of the inertia tensor (max and min).

    Parameters:
        geoms (gpd.GeoDataFrame): A GeoDataFrame containing building footprint geometries as polygons.

    Returns:
        list: A list with the same order as geoms which contains the calculated Polsby-Popper index for each geometry.
    """
    geoms = geoms.copy() 
    if type(geoms) is gpd.GeoSeries:
        geoms = gpd.GeoDataFrame({},geometry=geoms,crs=geoms.crs)
            
    # Ensure the geometries are in a projected CRS for accurate area and length calculations
    if not geoms.crs.is_projected:
        geoms = geoms.to_crs(geoms.geometry.estimate_utm_crs())

    I_max, I_min = calc_inertia_principal(geoms.geometry,principal_dirs=False)
    return list(np.sqrt(I_min / I_max))

def circunsribed_slenderness(geoms:gpd.GeoDataFrame|gpd.GeoSeries) -> list:
    geoms = geoms.copy() 
    if type(geoms) is gpd.GeoSeries:
        geoms = gpd.GeoDataFrame({},geometry=geoms,crs=geoms.crs)
            
    # Ensure the geometries are in a projected CRS for accurate area and length calculations
    if not geoms.crs.is_projected:
        geoms = geoms.to_crs(geoms.geometry.estimate_utm_crs())

    inertia_df = calc_inertia_principal(geoms, principal_dirs=True)

    total_length_1, total_length_2 = circunscribed_rectangle(
        geoms.geometry,
        inertia_df[1][:,0],
        inertia_df[1][:,1],
        inertia_df[3][:,0],
        inertia_df[3][:,1],
        return_length=True
    )

    return list(np.maximum(np.array(total_length_1) / np.array(total_length_2),
                  np.array(total_length_2) / np.array(total_length_1)))   

def min_bbox_slenderness(geoms:gpd.GeoDataFrame|gpd.GeoSeries):
    geoms = geoms.copy() 
    if type(geoms) is gpd.GeoSeries:
        geoms = gpd.GeoDataFrame({},geometry=geoms,crs=geoms.crs)
            
    # Ensure the geometries are in a projected CRS for accurate area and length calculations
    if not geoms.crs.is_projected:
        geoms = geoms.to_crs(geoms.geometry.estimate_utm_crs())

    L1, L2 = min_bbox(geoms,return_length=True)
    return list(np.maximum(np.array(L1) / np.array(L2),
              np.array(L2) / np.array(L1)))  

def inertia_circle(geoms:gpd.GeoDataFrame|gpd.GeoSeries) -> list:
    """
    Calculates the inertia irregularity index for building footprint polygons comparing inertia with the inertia of a circle with the same area.

    The inertia irregularity index is a measure of shape compactness, defined by the formula:
        inertia irregularity index = eq circle inertia / polygon inertia

    Parameters:
        geoms (gpd.GeoDataFrame): A GeoDataFrame containing building footprint geometries as polygons.

    Returns:
        list: A list with the same order as geoms which contains the calculated Polsby-Popper index for each geometry.
    """
    geoms = geoms.copy()
    if type(geoms) is gpd.GeoSeries:
        geoms = gpd.GeoDataFrame({},geometry=geoms,crs=geoms.crs)
    
    # Ensure the geometries are in a projected CRS for accurate area and length calculations
    if not geoms.crs.is_projected:
        geoms = geoms.to_crs(geoms.geometry.estimate_utm_crs())
    
    # Calculate the inertia irregularity score comparing to a circle with the same area
    geoms['inertia_circle'] = eq_circle_intertia(geoms.geometry.area) / np.abs(calc_inertia_z(geoms.geometry)) 
    
    return list(geoms['inertia_circle'])

def compactness(geoms:gpd.GeoDataFrame|gpd.GeoSeries) -> list:
    geoms = geoms.copy()
    if type(geoms) is gpd.GeoSeries:
        geoms = gpd.GeoDataFrame({},geometry=geoms,crs=geoms.crs)
            
    # Ensure the geometries are in a projected CRS for accurate area and length calculations
    if not geoms.crs.is_projected:
        geoms = geoms.to_crs(geoms.geometry.estimate_utm_crs())
        
    convex_hull = geoms.geometry.convex_hull
    geoms_holes_filled = geoms.geometry.apply(
        lambda x: Polygon(x.exterior)
    )
    return list(1 - (convex_hull.area - geoms_holes_filled.area) / convex_hull.area)

def eurocode_8_eccentricity_ratio(geoms:gpd.GeoDataFrame|gpd.GeoSeries) -> list:
    ratio = eurocode_8_df(geoms) 
    return list(ratio['eccentricity_ratio'])

def eurocode_8_eccentricity_ratio(geoms:gpd.GeoDataFrame|gpd.GeoSeries) -> list:
    ratio = eurocode_8_df(geoms) 
    return list(ratio['radius_ratio'])
    
def eurocode_8_slenderness(geoms:gpd.GeoDataFrame|gpd.GeoSeries) -> list:
    return inertia_slenderness(geoms) 

def eurocode_8_compactness(geoms:gpd.GeoDataFrame|gpd.GeoSeries) -> list:
    return compactness(geoms) 
    
def eurocode_8_df(geoms:gpd.GeoDataFrame|gpd.GeoSeries) -> pd.DataFrame:
    import scipy 

    geoms = geoms.copy()
    if type(geoms) is gpd.GeoSeries:
        geoms = gpd.GeoDataFrame({},geometry=geoms,crs=geoms.crs)
            
    # Ensure the geometries are in a projected CRS for accurate area and length calculations
    if not geoms.crs.is_projected:
        geoms = geoms.to_crs(geoms.geometry.estimate_utm_crs())

    # Compute principal moments of inertia and their corresponding eigenvectors
    inertia_df = calc_inertia_principal(geoms, principal_dirs=True)

    geoms = gpd.GeoDataFrame({},geometry=geoms,crs=geoms.crs)
    geoms['center_of_mass'] = center_of_mass(geoms)
    geoms['center_of_stiffness'] = geoms.geometry.boundary.centroid
    # Compute eccentricity vectors (difference between centroid and boundary centroid)
    e_vect = geoms.geometry.apply(lambda geom: np.array([
        geom['center_of_mass'].x - geom['center_of_stiffness'].x,
        geom['center_of_mass'].y - geom['center_of_stiffness'].y
    ]), axis=1)
    geoms = geoms.geometry
    # Compute magnitude of eccentricity vectors
    e_magnitude = np.sqrt(np.sum(np.array([*(e_vect * e_vect)]), axis=1))

    # Compute the area of the footprint in m2 
    area = geoms.geometry.to_crs(geoms.geometry.estimate_utm_crs()).area
    
    # Create DataFrame with necessary parameters
    df = pd.DataFrame({
        'e_vect': e_vect,
        'e_magnitude': e_magnitude,
        'area' : area,
        'I_1': inertia_df[0],  # First principal moment of inertia
        'I_2': inertia_df[2],  # Second principal moment of inertia
        'I_0' : inertia_df[0] + inertia_df[2], # Polar inertia 
        'I_t' : (inertia_df[0] + inertia_df[2]) + area * e_magnitude ** 2, # Torsional inertia
        'r' : 0.5 * (inertia_df[0] - inertia_df[2]), #Mohr radius 
        'c' : 0.5 * (inertia_df[0] + inertia_df[2]), #Mohr centre
        'vect_1': [row for row in inertia_df[1]],  # First principal axis
        'vect_2': [row for row in inertia_df[3]],  # Second principal axis
    })

    # Compute angle `b`. Angle of eccentricity direction and principal axis
    df['b'] = df.apply(
        lambda row: 0 if row['e_magnitude'] <= 10**-10 else get_angle(row['vect_1'], row['e_vect']),
        axis=1
    )

    # Optimize for the angle 'x' with the worst ecentricity ratio.
    df['x_opt'] = df.apply(
        lambda row: 0 if row['e_magnitude'] <= 10**-10 else scipy.optimize.fmin(
            lambda x: - np.cos(x - row['b']) ** 2 * (row['c'] - row['r'] * np.cos(-2 * x)),
            x0=0,
            xtol=1e-5,
            ftol=1e-5,
            disp=False
        )[0],
        axis=1
    )

    torsional_radius = np.sqrt(df['I_t'] / (df['c'] - df['r'] * np.cos(-2 * df['x_opt'])))

    radius_of_gyration = np.sqrt(df['I_0']/df['area'])

    eccentricity_ratio = df['e_magnitude'] * np.abs(np.cos(df['x_opt'] - df['b'])) / torsional_radius

    radius_ratio = torsional_radius / radius_of_gyration
                      
    slenderness_result = np.sqrt(df['I_1'] / df['I_2'])

    compactness_result = compactness(geoms) 

    vect_1 = np.array([*df['vect_1']])
    angle_vect_1 = np.arctan2(vect_1[:,1],vect_1[:,0]) 
    angle_eccentricity = np.abs(angle_vect_1 + df['x_opt'] + np.pi/2) # facing north
    angle_eccentricity[angle_eccentricity > 2*np.pi] -= 2*np.pi
    angle_eccentricity[angle_eccentricity > np.pi/2] -= np.pi 
    angle_eccentricity *= -180 / np.pi # invert to rotate north-east

    angle_slenderness = np.abs(angle_vect_1 + np.pi/2) # facing north 
    angle_slenderness[angle_slenderness > 2*np.pi] -= 2*np.pi
    angle_slenderness[angle_slenderness > np.pi/2] -= np.pi 
    angle_slenderness *= -180 / np.pi # invert to rotate north-east

    result_df = pd.DataFrame({
        'eccentricity_ratio':eccentricity_ratio,
        'radius_ratio':radius_ratio,
        'slenderness':slenderness_result,
        'compactness':compactness_result,
        'angle_eccentricity':angle_eccentricity,
        'angle_slenderness':angle_slenderness
    })
    result_df.index = geoms.index
    return result_df


def costa_rica_eccentricity_ratio(geoms:gpd.GeoDataFrame|gpd.GeoSeries) -> list:
    ratio = codigo_sismico_costa_rica_df(geoms) 
    return list(ratio['eccentricity_ratio'])
    
def codigo_sismico_costa_rica_df(geoms:gpd.GeoDataFrame|gpd.GeoSeries) -> pd.DataFrame:
    import scipy 
               
    geoms = geoms.copy()
    if type(geoms) is gpd.GeoSeries:
        geoms = gpd.GeoDataFrame({},geometry=geoms,crs=geoms.crs)
            
    # Ensure the geometries are in a projected CRS for accurate area and length calculations
    if not geoms.crs.is_projected:
        geoms = geoms.to_crs(geoms.geometry.estimate_utm_crs())
             
    # Compute principal moments of inertia and their corresponding eigenvectors
    inertia_df = calc_inertia_principal(geoms, principal_dirs=True)
    
    geoms = gpd.GeoDataFrame({},geometry=geoms,crs=geoms.crs)
    geoms['center_of_stiffness'] = geoms.geometry.centroid
    geoms['center_of_mass'] = center_of_mass(geoms)
    
    # Compute eccentricity vectors (difference between centroid and boundary centroid)
    e_vect = geoms.geometry.apply(lambda geom: np.array([
        geoms['center_of_mass'].x - geom['center_of_stiffness'].x,
        geoms['center_of_mass'].y - geom['center_of_stiffness'].y
    ]), axis=1)
    geoms = geoms.geometry
    
    # Compute magnitude of eccentricity vectors
    e_magnitude = np.sqrt(np.sum(np.array([*(e_vect * e_vect)]), axis=1))

    # Compute the area of the footprint in m2 
    area = geoms.geometry.to_crs(geoms.geometry.estimate_utm_crs()).area
    
    # Create DataFrame with necessary parameters
    df = pd.DataFrame({
        'e_vect': e_vect,
        'e_magnitude': e_magnitude,
        'area' : area,
        'I_1': inertia_df[0],  # First principal moment of inertia
        'I_2': inertia_df[2],  # Second principal moment of inertia
        'r' : 0.5 * (inertia_df[0] - inertia_df[2]), #Mohr radius 
        'c' : 0.5 * (inertia_df[0] + inertia_df[2]), #Mohr centre
        'vect_1': [row for row in inertia_df[1]],  # First principal axis
        'vect_2': [row for row in inertia_df[3]],  # Second principal axis
    })

    # Compute angle `b`. Angle of eccentricity direction and principal axis
    df['b'] = df.apply(
        lambda row: 0 if row['e_magnitude'] <= 10**-10 else get_angle(row['vect_1'], row['e_vect']),
        axis=1
    )

    # Optimize for the angle 'x' with the worst ecentricity ratio.
    df['x_opt'] = df.apply(
        lambda row: 0 if row['e_magnitude'] <= 10**-10 else scipy.optimize.fmin(
            lambda x: - np.cos(x - row['b']) ** 4 *  (
                    row['c'] - row['r'] * np.cos(-2*x) ### Max inertia is in the min side and min inertia in the max length side
                        ) / (
                    row['c'] + row['r'] * np.cos(-2*x)
                ),
            x0=0,
            xtol=1e-5,
            ftol=1e-5,
            disp=False
        )[0],
        axis=1
    )

    eccentricity_i = np.abs(df['e_magnitude'] * np.cos(df['x_opt'] - df['b']))
    dimension_i = np.sqrt(df['area']) * ((df['c'] + df['r'] * np.cos(-2*df['x_opt'])) / (df['c'] - df['r'] * np.cos(-2*df['x_opt']))) ** 0.25
    eccentricity_ratio = eccentricity_i / dimension_i
    vect_1 = np.array([*df['vect_1']])
    angle = np.abs(np.arctan2(vect_1[:,1],vect_1[:,0]) + df['x_opt'] + np.pi/2) # facing north
    angle[angle > 2*np.pi] -= 2*np.pi 
    angle[angle > np.pi/2] -= np.pi
    angle *= -180/np.pi  # invert to rotate north-east

    result_df = pd.DataFrame({'eccentricity_ratio' : eccentricity_ratio, 'angle' : angle})
    result_df.index = geoms.index
         
    return result_df



def asce_7_setback_ratio(geoms:gpd.GeoDataFrame) -> list:
    geoms = geoms.copy()
    if type(geoms) is gpd.GeoSeries:
        geoms = gpd.GeoDataFrame({},geometry=geoms,crs=geoms.crs)
            
    # Ensure the geometries are in a projected CRS for accurate area and length calculations
    if not geoms.crs.is_projected:
        geoms = geoms.to_crs(geoms.geometry.estimate_utm_crs())
        
    convex_hull = geoms.geometry.convex_hull
    geoms_holes_filled = geoms.geometry.apply(
        lambda x: Polygon(x.exterior)
    )

    L1,b1, L2,b2 = basic_lengths(geoms,get_a=False)
    setback_ratio = list(np.minimum(np.array(b1) / np.array(L1),
              np.array(b2) / np.array(L2)))  
    
    return setback_ratio


def asce_7_hole_ratio(geoms:gpd.GeoDataFrame|gpd.GeoSeries) -> list:
    geoms = geoms.copy()
    if type(geoms) is gpd.GeoSeries:
        geoms = gpd.GeoDataFrame({},geometry=geoms,crs=geoms.crs)
            
    # Ensure the geometries are in a projected CRS for accurate area and length calculations
    if not geoms.crs.is_projected:
        geoms = geoms.to_crs(geoms.geometry.estimate_utm_crs())
        
    geoms_holes_filled = geoms.geometry.apply(
        lambda x: Polygon(x.exterior)
    )    

    df = gpd.GeoDataFrame(
        {
            'index':geoms.index,
            'polygon_with_holes':geoms.geometry,
            'polygon':geoms_holes_filled,
        },
        geometry = geoms.geometry.apply(
            lambda x: MultiPolygon([Polygon(ring) for ring in x.interiors]) if x.interiors else Polygon()
        ),
        crs = geoms.crs
    )
    df = df.loc[df.geometry.is_empty == False]
    df = df.explode().reset_index(drop=True)
    inertia_df = calc_inertia_principal(df.geometry,principal_dirs=True)
    inertia_df_vect_ids = [3,1]
         
    for i in range(2):
        id = inertia_df_vect_ids[i]
        
        df['distance'] = np.sqrt((
                df['polygon'].bounds['maxx']-df['polygon'].bounds['minx']
            )**2 + (
                df['polygon'].bounds['maxy']-df['polygon'].bounds['miny']
            )**2) / 2
        df['line_start_x'] = df.geometry.centroid.x - inertia_df[id][:,0] * (df['distance'] + 1) 
        df['line_start_y'] = df.geometry.centroid.y - inertia_df[id][:,1] * (df['distance'] + 1) 
        df['line_end_x'] = df.geometry.centroid.x + inertia_df[id][:,0] * (df['distance'] + 1) 
        df['line_end_y'] = df.geometry.centroid.y + inertia_df[id][:,1] * (df['distance'] + 1)    
        df['line'] = gpd.GeoSeries(df.apply(lambda row: LineString([(row['line_start_x'],row['line_start_y']),(row['line_end_x'],row['line_end_y'])]),axis=1),crs=df.crs)
        df['intersection'] = df['polygon'].intersection(df['line'])
        df = df.explode(column='intersection').reset_index(drop=True)
        df = df.loc[df['intersection'].distance(df.centroid) < 10**-3]
        df[f'side_length_{i+1}'] = df['intersection'].length
        df[f'hole_width_{i+1}_a'] = df[f'side_length_{i+1}'] - df['polygon_with_holes'].intersection(df['intersection']).length
        if i == 0:
            df[f'hole_width_{i+1}_b'] = np.sqrt(df.geometry.area * np.sqrt(inertia_df[2] / inertia_df[0]))
        else:
            df[f'hole_width_{i+1}_b'] = np.sqrt(df.geometry.area * np.sqrt(inertia_df[0] / inertia_df[2]))

        df[f'hole_width_{i+1}'] = df[[f'hole_width_{i+1}_a',f'hole_width_{i+1}_b']].max(axis=1)
        df[f'hole_ratio_{i+1}'] = df[f'hole_width_{i+1}'] / df[f'side_length_{i+1}']
    
    df['hole_ratio'] = df[['hole_ratio_1','hole_ratio_2']].max(axis=1)
    hole_ratio = df.loc[df.groupby('index')['hole_ratio'].idxmax(),['index','hole_ratio']]
    hole_ratio = geoms.merge(hole_ratio, left_index=True, right_on='index', how='left').fillna({'hole_ratio': 0})
    hole_ratio = list(hole_ratio['hole_ratio'])
    return hole_ratio


def asce_7_df(geoms:gpd.GeoDataFrame|gpd.GeoSeries) -> pd.DataFrame:
    geoms = geoms.copy()
    if type(geoms) is gpd.GeoSeries:
        geoms = gpd.GeoDataFrame({},geometry=geoms,crs=geoms.crs)
            
    # Ensure the geometries are in a projected CRS for accurate area and length calculations
    if not geoms.crs.is_projected:
        geoms = geoms.to_crs(geoms.geometry.estimate_utm_crs())
     
    setback_ratio_results = setback_ratio(geoms)
    hole_ratio_results = hole_ratio(geoms)

    result_df = pd.DataFrame({'setback_ratio':setback_ratio_results,'hole_ratio':hole_ratio_results})
    result_df.index = geoms.index
         
    return result_df

def NTC_mexico_df(geoms:gpd.GeoDataFrame|gpd.GeoSeries) -> pd.DataFrame:
    return asce_7_df(geoms)

def gndt_beta_1_main_shape_slenderness(geoms:gpd.GeoDataFrame|gpd.GeoSeries) -> list:
    geoms = geoms.copy()
    if type(geoms) is gpd.GeoSeries:
        geoms = gpd.GeoDataFrame({},geometry=geoms,crs=geoms.crs)
            
    # Ensure the geometries are in a projected CRS for accurate area and length calculations
    if not geoms.crs.is_projected:
        geoms = geoms.to_crs(geoms.geometry.estimate_utm_crs())

    geoms_holes_filled = geoms.geometry.apply(
        lambda x: Polygon(x.exterior)
    )  

    L1,a1, L2,a2 = basic_lengths(geoms_holes_filled,get_b=False)
    df = pd.DataFrame({
        'L1': L1, 'a1': a1,
        'L2': L2, 'a2': a2
    })
    
    # Choose values based on where a1 > a2
    df['L'] = df['L1'].where(df['a1'] > df['a2'], df['L2'])
    df['a'] = df['a1'].where(df['a1'] > df['a2'], df['a2'])
    
    # Extract the final lists
    L = df['L'].to_numpy()
    a = df['a'].to_numpy()

    beta_1 = list(a/L)
    return list(beta_1) 
    
def gndt_beta_2_setback_ratio(geoms:gpd.GeoDataFrame|gpd.GeoSeries) -> list:
    geoms = geoms.copy()
    if type(geoms) is gpd.GeoSeries:
        geoms = gpd.GeoDataFrame({},geometry=geoms,crs=geoms.crs)
            
    # Ensure the geometries are in a projected CRS for accurate area and length calculations
    if not geoms.crs.is_projected:
        geoms = geoms.to_crs(geoms.geometry.estimate_utm_crs())

    geoms_holes_filled = geoms.geometry.apply(
        lambda x: Polygon(x.exterior)
    )  

    L1,a1, L2,a2 = basic_lengths(geoms_holes_filled,get_b=False)
    df = pd.DataFrame({
        'L1': L1, 'a1': a1, 'b1': b1,
        'L2': L2, 'a2': a2, 'b2': b2
    })
    
    # Choose values based on where a1 > a2
    df['L'] = df['L1'].where(df['a1'] > df['a2'], df['L2'])
    df['a'] = df['a1'].where(df['a1'] > df['a2'], df['a2'])
    df['b'] = df['b1'].where(df['a1'] > df['a2'], df['b2'])
    
    # Extract the final lists
    L = df['L'].to_numpy()
    a = df['a'].to_numpy()
    b = df['b'].to_numpy()
    
    beta_2 = list(b/L)
    return list(beta_2)
    
def gndt_beta_3_footprint_slenderness(geoms:gpd.GeoDataFrame|gpd.GeoSeries) -> list:
    return min_bbox_slenderness(geoms)
    
def gndt_beta_4_eccentricity_ratio(geoms:gpd.GeoDataFrame|gpd.GeoSeries,directions:tuple=None,main_shape_lengths:tuple=None) -> list:
    geoms = geoms.copy()
    if type(geoms) is gpd.GeoSeries:
        geoms = gpd.GeoDataFrame({},geometry=geoms,crs=geoms.crs)
            
    # Ensure the geometries are in a projected CRS for accurate area and length calculations
    if not geoms.crs.is_projected:
        geoms = geoms.to_crs(geoms.geometry.estimate_utm_crs())

    geoms_holes_filled = geoms.geometry.apply(
        lambda x: Polygon(x.exterior)
    )  

    if directions is not None:
        dir_1_x, dir_1_y, dir_2_x, dir_2_y = directions 
    else:
        rectangles = shapely.minimum_rotated_rectangle(geoms.geometry)
        rectangles = gpd.GeoSeries(rectangles,crs=geoms.crs)
        dir_1_x, dir_1_y, dir_2_x, dir_2_y = rectangle_to_directions(rectangles,normalize=True)
    
    if main_shape_lengths is not None:
        a1,a2 = main_shape_lengths
    else:
        L1,a1, L2,a2 = basic_lengths(geoms_holes_filled,get_b=False)

    geoms['center_of_stiffness'] = geoms.geometry.boundary.centroid
    geoms['center_of_mass'] = center_of_mass(geoms)
    # Compute eccentricity vectors (difference between centroid and boundary centroid)
    e_x = geoms.geometry.apply(lambda geom: np.array(
        geoms['center_of_mass'].x - geom['center_of_stiffness'].x
    ), axis=1)
    
    e_y = geoms.geometry.apply(lambda geom: np.array(
        geoms['center_of_mass'].y - geom['center_of_stiffness'].y
    ), axis=1)

    e_1 = dir_1_x * e_x + dir_1_y * e_y
    e_2 = dir_2_x * e_x + dir_2_y * e_y
    
    df = pd.DataFrame({
        'e1': e1, 'a1': a1,
        'e2': e2, 'a2': a2
    })
    
    # Choose values based on where a1 > a2
    df['e'] = df['e1'].where(df['a1'] > df['a2'], df['e2'])
    df['a'] = df['a1'].where(df['a1'] > df['a2'], df['a2'])
    
    # Extract the final lists
    e = df['e'].to_numpy()
    a = df['a'].to_numpy()
    
    beta_4 = list(e/a)
    return list(beta_4)
    
def gndt_beta_6_setback_slenderness(geoms:gpd.GeoDataFrame|gpd.GeoSeries,directions:tuple=None,setback_lengths:tuple=None) -> list:
    geoms = geoms.copy()
    if type(geoms) is gpd.GeoSeries:
        geoms = gpd.GeoDataFrame({},geometry=geoms,crs=geoms.crs)
            
    # Ensure the geometries are in a projected CRS for accurate area and length calculations
    if not geoms.crs.is_projected:
        geoms = geoms.to_crs(geoms.geometry.estimate_utm_crs())

    geoms_holes_filled = geoms.geometry.apply(
        lambda x: Polygon(x.exterior)
    )  

    if directions is not None:
        dir_1_x, dir_1_y, dir_2_x, dir_2_y = directions
    else:
        rectangles = shapely.minimum_rotated_rectangle(geoms.geometry)
        rectangles = gpd.GeoSeries(rectangles,crs=geoms.crs)
        dir_1_x, dir_1_y, dir_2_x, dir_2_y = rectangle_to_directions(rectangles,normalize=True)

    setbacks = gpd.GeoDataFrame({},geometry=geoms.geometry.convex_hull.difference(geoms_holes_filled.geometry),crs=geoms.crs)
    df = gpd.GeoDataFrame(
        {
            'index':geoms.index,
            'polygon_with_holes':geoms.geometry,
            'polygon':geoms_holes_filled
        },
        geometry = setbacks,
        crs = geoms.crs
    )
    df = df.loc[df.geometry.is_empty == False]
    df = df.explode().reset_index(drop=True)

    if setbacks_lengths is not None:
        b1,b2 = setbacks_lengths
    else:
        b1, b2 = circunscribed_setback_length(geoms)

    dir_x = [dir_1_x,dir_2_x]
    dir_y = [dir_1_y,dir_2_y]
    b = [b1,b2]
    for i in range(2):        
        df['distance'] = np.sqrt((
                df['polygon'].bounds['maxx']-df['polygon'].bounds['minx']
            )**2 + (
                df['polygon'].bounds['maxy']-df['polygon'].bounds['miny']
            )**2) / 2
        df['line_start_x'] = df.geometry.centroid.x - dir_x[i] * (df['distance'] + 1) 
        df['line_start_y'] = df.geometry.centroid.y - dir_y[i] * (df['distance'] + 1) 
        df['line_end_x'] = df.geometry.centroid.x + dir_x[i] * (df['distance'] + 1) 
        df['line_end_y'] = df.geometry.centroid.y + dir_y[i] * (df['distance'] + 1)    
        df['line'] = gpd.GeoSeries(df.apply(lambda row: LineString([(row['line_start_x'],row['line_start_y']),(row['line_end_x'],row['line_end_y'])]),axis=1),crs=df.crs)
        df['intersection'] = df['polygon'].intersection(df['line'])
        df = df.explode(column='intersection').reset_index(drop=True)
        df = df.loc[df['intersection'].distance(df.centroid) < 10**-3]
        df[f'c_{i+1}'] = df['intersection'].length
        df[f'b_{i+1}'] = b[i]

        df[f'setback_slenderness_{i+1}'] = df[f'c_{i+1}'] / df[f'b_{i+1}']
    
    df['setback_slenderness'] = df[['setback_slenderness_1','setback_slenderness_2']].max(axis=1)
    setback_slenderness = df.loc[df.groupby('index')['setback_slenderness'].idxmax(),['index','setback_slenderness']]
    setback_slenderness = geoms.merge(setback_slenderness, left_index=True, right_on='index', how='left').fillna({'setback_slenderness': 0})
    setback_slenderness = list(setback_slenderness['setback_slenderness'])
    return setback_slenderness

def gndt_italy_df(geoms:gpd.GeoDataFrame) -> pd.DataFrame:
    geoms = geoms.copy()
    if type(geoms) is gpd.GeoSeries:
        geoms = gpd.GeoDataFrame({},geometry=geoms,crs=geoms.crs)
            
    # Ensure the geometries are in a projected CRS for accurate area and length calculations
    if not geoms.crs.is_projected:
        geoms = geoms.to_crs(geoms.geometry.estimate_utm_crs())

    geoms_holes_filled = geoms.geometry.apply(
        lambda x: Polygon(x.exterior)
    )    
     
    L1,a1,b1, L2,a2,b2 = basic_lengths(geoms_holes_filled)
    df = pd.DataFrame({
        'L1': L1, 'a1': a1, 'b1': b1,
        'L2': L2, 'a2': a2, 'b2': b2
    })
    
    # Choose values based on where a1 > a2
    df['L'] = df['L1'].where(df['a1'] > df['a2'], df['L2'])
    df['a'] = df['a1'].where(df['a1'] > df['a2'], df['a2'])
    df['b'] = df['b1'].where(df['a1'] > df['a2'], df['b2'])
    
    # Extract the final lists
    L = df['L'].to_numpy()
    a = df['a'].to_numpy()
    b = df['b'].to_numpy()

    beta_1 = list(a/L)
    beta_2 = list(b/L)
    beta_3 = list(np.maximum(np.array(L1) / np.array(L2),
              np.array(L2) / np.array(L1)))  

    rectangles = shapely.minimum_rotated_rectangle(geoms.geometry)
    rectangles = gpd.GeoSeries(rectangles,crs=geoms.crs)
    dir_1_x, dir_1_y, dir_2_x, dir_2_y = rectangle_to_directions(rectangles,normalize=True)
    
    beta_4 = gndt_beta_4_eccentricity_ratio(geoms,directions=(dir_1_x, dir_1_y, dir_2_x, dir_2_y),main_shape_lengths=(a1,a2))
    beta_6 = gndt_beta_6_setback_slenderness(geoms,directions=(dir_1_x, dir_1_y, dir_2_x, dir_2_y),setback_lengths=(b1,b2))

    return pd.DataFrame({'beta_1_main_shape_slenderness':beta_1,'beta_2_setback_ratio':beta_2, 'beta_3_footprint_slenderness':beta_3, 'beta_4_eccentricity_ratio':beta_4, 'beta_6_setback_slenderness':beta_6})
    

    
