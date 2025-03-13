import geopandas as gpd 
import pandas as pd
import shapely 
import numpy as np
import warnings
from .utils import get_normal, explode_edges, calculate_momentum, select_touching_edges, resultant_angle, calc_inertia_z

def contact_forces_df(geoms:gpd.GeoDataFrame,buffer:float=0,height_column:str=None,min_radius:float=0) -> pd.DataFrame:
    """
    Calculates force-based metrics for building footprints based on their geometry and proximity.

    Parameters:
        geoms (gpd.GeoDataFrame): A GeoDataFrame containing building footprints as polygon geometries.
        buffer (float): Buffer distance in meters to determine if two buildings are considered touching.
        height_column (str, optional): Column name in `geoms` specifying the building height in meters.
                                       If `None`, all buildings are assumed to have a uniform height of 1.
        min_radius (float, optional): Minimum distance multiplier used to exclude forces that would otherwise 
                                        increase momentum. Forces with a distance below a threshold 
                                        (`min_radius * equivalent radius`) will contribute to the momentum 
                                        calculation only if they decrease the momentum. The equivalent radius 
                                        of a building is defined as the radius of a circle with the same area 
                                        as the building's footprint.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame with the same rows as geoms and the following columns:
            - "geometry": Geometry column of type `shapely.Polygon`.
            - "height": The building height. If `height_column` is `None` the height will be 1.
            - "angular_acc": Angular acceleration calculated as (momentum * area / inertia). Momentum of the resultant force with respect to the footprint centroid.
              Formula for momentum: sum(distance * |force_i|) Formula for angular acceleration: momentum * area / inertia
            - "force": Magnitude of the resultant force acting on the footprint. Normalized (divided) by the sqrt of the area. 
              Formula: |sum(force_i)|
            - "confinement_ratio": Proportion of total forces that are confined (counterbalanced by opposing forces).
              Formula: (sum(|force_i|) - |sum(force_i)|) / |sum(force_i)|
            - "angle": Normalized sum of the angles between individual forces and the resultant force.
              Formula: sum(|force_i| * angle(force_i, sum(force_j))) / |sum(force_i)|

    Notes:
        - This method assumes that forces are calculated based on proximity and building height.
        - The buffer determines which buildings are close enough to interact.
    """

    #if "force" in geoms.columns:
    #    warnings.warn("The 'force' column already exists and will be overwritten.", UserWarning)
        
    #if "confinement_ratio" in geoms.columns:
    #    warnings.warn("The 'confinement_ratio' column already exists and will be overwritten.", UserWarning)
        
    #if "angular_acc" in geoms.columns:
    #    warnings.warn("The 'angular_acc' column already exists and will be overwritten.", UserWarning)
        
    #if "angle" in geoms.columns:
    #    warnings.warn("The 'angle' column already exists and will be overwritten.", UserWarning)

    #geoms = geoms.drop(columns=['angle','angular_acc','confinement_ratio','force'],errors='ignore')

    geoms = geoms.copy()
    geoms_orig = geoms[[geoms.geometry.name]].copy()
    orig_crs = geoms.crs
    geoms['geom_id'] = geoms.index.copy()
    geoms.geometry = geoms.geometry.force_2d()

    if type(height_column) == type(None):
        geoms['height'] = 1
    else:
        geoms['height'] = geoms[height_column].astype(float)

    if not geoms.crs.is_projected:
        geoms = geoms.to_crs(geoms.geometry.estimate_utm_crs())

    geoms['inertia'] = calc_inertia_z(geoms.geometry)
    geoms['area'] = geoms.geometry.area
    crs = geoms.crs
    geoms['centroid'] = geoms.geometry.centroid.copy()

    geoms = select_touching_edges(geoms,buffer=buffer)
    geoms = explode_edges(geoms,min_length=buffer)

    geoms[['edge_center','normal_vector']] = geoms.apply(lambda x: pd.Series(get_normal(x['edges'],scale=x['height'])),axis=1)

    geoms['momentum'] = geoms.apply(lambda x:calculate_momentum(x['edge_center'],x['normal_vector'],x['centroid'],min_dist=min_radius*np.sqrt(x['area']/np.pi)),axis=1)

    geoms['abs_force'] = geoms.apply(lambda x: np.sqrt(x['normal_vector'][0]**2+x['normal_vector'][1]**2),axis=1)

    geoms = resultant_angle(geoms,vector_column='normal_vector',id_column='geom_id')

    geoms['angle'] = geoms['angle'] * 2
    geoms['angle'] = geoms['angle'] * geoms['abs_force']

    geoms = geoms.groupby('geom_id').agg({
        'height':'first',
        'normal_vector':'sum',
        'abs_force':'sum',
        'angle':'sum',
        'momentum':'sum',
        'inertia':'first',
        'area':'first'
    })

    geoms = geoms.rename(columns={'normal_vector':'res_normal'})

    geoms['force'] = geoms.apply(lambda x: np.sqrt(x['res_normal'][0]**2 + x['res_normal'][1]**2),axis=1)
    geoms['confinement_ratio'] = (geoms['abs_force'] - geoms['force']) / geoms['force']
    geoms['momentum'] = np.abs(geoms['momentum']).apply(min)
    geoms['angle'] = geoms['angle'] / geoms['abs_force']
    geoms['angular_acc'] = geoms['momentum'] / geoms['inertia']

    geoms['force'] = geoms['force'] / np.sqrt(geoms['area'])
    geoms['angular_acc'] = geoms['angular_acc'] * geoms['area']

    geoms = geoms_orig.merge(geoms[['height','force','confinement_ratio','angular_acc','angle']],left_index=True,right_index=True,how='left')
    geoms = geoms[['height','force','confinement_ratio','angular_acc','angle',geoms.geometry.name]]
    geoms.loc[geoms['force'].isna(),'force'] = 0 
    geoms.loc[geoms['confinement_ratio'].isna(),'confinement_ratio'] = 0 
    geoms.loc[geoms['angular_acc'].isna(),'angular_acc'] = 0 
    geoms.loc[geoms['angle'].isna(),'angle'] = 0 
    #geoms[['height','force','confinement_ratio','angular_acc','angle']] = geoms[['height','force','confinement_ratio','angular_acc','angle']].astype(float)
    geoms = geoms[['height','force','confinement_ratio','angular_acc','angle']].astype(float)

    return geoms

def relative_position(
    forces: gpd.GeoDataFrame|pd.DataFrame,
    min_angular_acc: float = 2.133,
    min_confinement: float = 1,
    min_angle: float = 0.78,
    min_force: float = 0.166,
    buffer:float=0,
    height_column:str=None,
    min_radius:float=0
) -> list:
    """
    Classifies building footprints based on their relative position and interaction with surrounding structures.

    Parameters:
        footprints (gpd.GeoDataFrame): GeoDataFrame outputted by `get_forces_gdf()` containing `force`, `confinement_ratio`, and `angle` columns.
        min_force (float, optional): Threshold for the resultant force (nomalized by the sqrt of the area).
                                     Default: 0.166 (e.g., for a square building with height 1 and side length 1, 
                                     a touching structure covering 1/6 of one side would have a resultant force of 1/6).
        min_angle (float, optional): Angle threshold in radians. Default: pi / 4 (45 degrees).
        min_confinement (float, optional): Threshold for confinement_ratio. Default: 1 (equal amounts of confined and resultant forces).
        min_angular_acc (float, optional): Threshold for angular acceleration (momentum * area / inertia). Default: 2.133
                                        (e.g., for a rectangular building with height 1 and sides of length 1 and 0.5, 
                                        a touching structure covering 1/3 of two sides in the worst case 
                                        would have an anuglar acceleration of 2.133).

    Returns:
        list: A list classifying buildings (same order as footprints geodataframe) into the following categories (priority order):
            1. "torque": Buildings classified as *confined* or *corner* with momentum exceeding the minimum momentum.
            2. "confined": Structures touching on both the left and right lateral sides.
            3. "corner": Structures touching at a corner, based on force and angle thresholds.
            4. "lateral": Structures touching on either the left or right side.
            5. "isolated": Structures with no touching neighbors.

    Notes:
        - The classification prioritizes categories in the order listed above.
    """
    forces = forces.copy()
    if (
        ('force' not in forces.columns) | 
        ('angle' not in forces.columns) | 
        ('confinement_ratio' not in forces.columns) | 
        ('angular_acc' not in forces.columns)
    ):
        forces = forces_df(forces,buffer=buffer,height_column=height_column,min_radius=min_radius)

    # Initialize 'relative_position' column
    forces['relative_position'] = 'isolated'
    
    # Update 'relative_position' based on criteria
    forces.loc[forces['force'] > min_force, 'relative_position'] = 'lateral'
    
    forces.loc[
        (
            forces['angle'] > min_angle
        ) & (
            forces['relative_position'] == 'lateral'
        ),
        'relative_position'
    ] = 'corner'
    
    forces.loc[forces['confinement_ratio'] > min_confinement, 'relative_position'] = 'confined'

    forces.loc[
        ((forces['relative_position'] == 'corner') | (forces['relative_position'] == 'confined')
        ) & (
            forces['angular_acc'] > min_angular_acc
        ), 
        'relative_position'
    ] = 'torque'
    
    return list(forces['relative_position'])
