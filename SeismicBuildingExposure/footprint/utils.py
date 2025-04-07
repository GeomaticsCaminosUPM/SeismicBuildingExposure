import  geopandas as gpd 
import pandas as pd
import shapely 
import numpy as np
from shapely.geometry import LineString, Point
from packaging.version import Version

"""
def get_angle_sharp(normal_0,normal_1,geom_id_0,geom_id_1):
    if geom_id_0 != geom_id_1:
        return 0 
     
    _normal_0 = normal_0 / np.sqrt(np.sum(normal_0 ** 2))
    _normal_1 = normal_1 / np.sqrt(np.sum(normal_1 ** 2))

    # Compute the dot and cross products
    dot = np.dot(_normal_0, _normal_1)  # Dot product

    if np.abs(dot) > 1:
        if np.abs(dot) > 1.0000001:
            warnings.warn(f"invalid value encountered in np.arccos({dot})", RuntimeWarning)

        dot = 1 * dot/np.abs(dot)

    cross = np.cross(_normal_0, _normal_1)  # Cross product (2D equivalent)
    
    if cross < 0:
        return 0 

    if np.abs(cross) > 1:
        if np.abs(cross) > 1.0000001:
            warnings.warn(f"invalid value encountered in np.arcsin({cross})", RuntimeWarning)

        cross = 1 * cross/np.abs(cross)

    angle = np.arccos(np.abs(dot))
    return angle
"""


def get_angle_90(normal_0,normal_1,geom_id_0=0,geom_id_1=0):
    if geom_id_0 != geom_id_1:
        return 0 

    l_normal_0 = np.sqrt(np.sum(normal_0 ** 2))
    l_normal_1 = np.sqrt(np.sum(normal_1 ** 2))
     
    if (l_normal_0 == 0) or (l_normal_1 == 0):
        return 0

    _normal_0 = normal_0 / l_normal_0
    _normal_1 = normal_1 / l_normal_1
    dot = _normal_0[0] * _normal_1[0] + _normal_0[1] * _normal_1[1]
    if np.abs(dot) > 1:
        if np.abs(dot) > 1.0000001:
            warnings.warn(f"invalid value encountered in np.arccos({np.abs(dot)})", RuntimeWarning)

        dot = 1

    angle = np.arccos(np.abs(dot))

    return angle


def get_angle(vect_0, vect_1, geom_id_0=0, geom_id_1=0):
    # If geometry IDs are different, the angle is undefined (return 0)
    if geom_id_0 != geom_id_1:
        return 0
    
    # Normalize the input vectors
    _vect_0 = vect_0 / np.linalg.norm(vect_0)
    _vect_1 = vect_1 / np.linalg.norm(vect_1)
    
    # Compute the dot and cross products
    dot = np.dot(_vect_0, _vect_1)  # Dot product
    if np.abs(dot) > 1:
        if np.abs(dot) > 1.0000001:
            warnings.warn(f"invalid value encountered in np.arccos({dot})", RuntimeWarning)

        dot = 1 * dot/np.abs(dot)

    cross = _vect_0[0] * _vect_1[1] - _vect_0[1] * _vect_1[0]
    #cross = np.cross(_vect_0, _vect_1)  # Cross product (2D equivalent)
    
    if np.abs(cross) > 1:
        if np.abs(cross) > 1.0000001:
            warnings.warn(f"invalid value encountered in np.arcsin({cross})", RuntimeWarning)

        cross = 1 * cross/np.abs(cross)

    # Calculate the angle using arctan2 for accurate quadrant handling
    angle = np.arctan2(cross, dot)
    
    return angle


def linestring_to_multilinestring(linestring):
    return shapely.MultiLineString(
        [
            LineString(
                (Point(linestring.coords[i]),
                Point(linestring.coords[i+1]))
            ) for i in range(len(linestring.coords)-1)
        ]
    )

def select_touching_edges(gdf,buffer=0):
    gdf_copy = gdf.copy()
    crs = gdf_copy.geometry.crs
    gdf_union = shapely.buffer(gdf_copy.geometry.union_all(),max(buffer,0.0001),cap_style='square',join_style='mitre')
    gdf_union = shapely.buffer(gdf_union,-buffer-0.001,cap_style='square',join_style='mitre')

    gdf_copy.geometry = gdf_copy.geometry.buffer(max(buffer,0.001),cap_style='square',join_style='mitre')
    gdf_copy.geometry = gdf_copy.geometry.buffer(min(-buffer,-0.001),cap_style='square',join_style='mitre')
    gdf_copy.geometry = gdf_copy.geometry.boundary.intersection(gdf_union)

    return gdf_copy


def explode_edges(gdf,min_length=0):
    gdf_copy = gdf.copy()
    crs = gdf.crs
    gdf_copy = gdf_copy.loc[gdf_copy.geometry.is_empty == False]
    gdf_copy = gdf_copy.explode(index_parts=False).reset_index(drop=True)
    gdf_copy['edges'] = gdf_copy.apply(lambda x: linestring_to_multilinestring(x['geometry']),axis=1)
    gdf_copy = gdf_copy.set_geometry('edges',crs=crs).explode().reset_index(drop=True)
    gdf_copy = gdf_copy.loc[gdf_copy['edges'].length > max(min_length,0.001),:]
    return gdf_copy


# Function to get normal vector at the center of a LineString with scaling factor x
def get_normal(linestring, scale=1):
    # 1. Get center point (by length interpolation)
    center_point = linestring.interpolate(0.5, normalized=True)
    
    # 2. Find the two closest points around the center for the tangent
    p1 = Point(linestring.coords[0])
    p2 = Point(linestring.coords[1])
    if len(linestring.coords) > 2:
        raise Exception("Linestring has more than 2 points.")
    # Tangent vector between p1 and p2
    tangent_vector = np.array([p2.x - p1.x, p2.y - p1.y])
    
    # 3. Compute normal vector (perpendicular to tangent)
    normal_vector = np.array([-tangent_vector[1], tangent_vector[0]])
    normal_vector /= np.sqrt(np.sum(normal_vector**2))
    
    # 4. Scale the normal vector
    length_of_linestring = linestring.length
    if scale == 0:
        scaled_normal_vector = normal_vector
    else: 
        scaled_normal_vector = normal_vector * (scale * length_of_linestring)
    
    return center_point, scaled_normal_vector


# Function to calculate momentum relative to a reference point
def calculate_momentum(center_point, normal_vector, reference_point,min_dist=0):
    r = np.array([center_point.x - reference_point.x, center_point.y - reference_point.y])
    cross_product = float(r[0] * normal_vector[1] - r[1] * normal_vector[0])
    if np.sqrt(np.sum(r**2)) < min_dist:
        if cross_product > 0:
            return np.array([cross_product,0,cross_product, 0])
        else:
            return np.array([cross_product,0,0, cross_product])
    else:
        return np.array([cross_product,cross_product,cross_product, cross_product])

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


def _ring_inertia_z(polygon):
    """
    implementation of the moment of area for a single ring
    """

    coordinates = shapely.get_coordinates(polygon)
    centroid = shapely.centroid(polygon)
    centroid_coords = shapely.get_coordinates(centroid)
    points = coordinates - centroid_coords
    return np.abs(sum(
        (points[:-1,0] * points[1:,1] - points[1:,0] * points[:-1,1]) * (
            points[1:,0]**2
            + points[1:,0] * points[:-1,0]
            + points[:-1,0]**2
            + points[1:,1]**2
            + points[1:,1] * points[:-1,1]
            + points[:-1,1]**2
        )
    ) / 12)


def calc_inertia_z(collection):
    # noqa: E501
    ga = _cast(collection)
    # construct a dataframe of the fundamental parts of all input polygons
    parts, collection_ix = shapely.get_parts(ga, return_index=True)
    rings, ring_ix = shapely.get_rings(parts, return_index=True)
    # get_rings() always returns the exterior first, then the interiors
    collection_ix = np.repeat(
        collection_ix, shapely.get_num_interior_rings(parts) + 1
    )
    # we need to work in polygon-space for the algorithms
    # (centroid, shoelace calculation) to work
    polygon_rings = shapely.polygons(rings)
    is_external = np.zeros_like(collection_ix).astype(bool)
    # the first element is always external
    is_external[0] = True
    # and each subsequent element is external iff
    # it is different from the preceeding index
    is_external[1:] = ring_ix[1:] != ring_ix[:-1]
    # now, our analysis frame contains a bunch of (guaranteed-to-be-simple) polygons
    # that represent either exterior rings or holes
    polygon_rings = gpd.GeoDataFrame(
        dict(
            collection_ix=collection_ix,
            ring_within_geom_ix=ring_ix,
            is_external_ring=is_external,
        ),
        geometry=polygon_rings,
    )
    # the polygonal moi can be calculated using the same ring-based strategy,
    # and this could be parallelized if necessary over the elemental shapes with:

    # from joblib import Parallel, parallel_backend, delayed
    # with parallel_backend('loky', n_jobs=-1):
    #     engine = Parallel()
    #     promise = delayed(_second_moment_of_area_polygon)
    #     result = engine(promise(geom) for geom in polygon_rings.geometry.values)

    # but we will keep simple for now
    polygon_rings["moa"] = polygon_rings.geometry.apply(_ring_inertia_z)
    # the above algorithm computes an unsigned moa
    # to be insensitive to winding direction.
    # however, we need to subtract the moa of holes. Hence, the sign of the moa is
    # -1 when the polygon is an internal ring and 1 otherwise:
    polygon_rings["sign"] = (1 - polygon_rings.is_external_ring * 2) * -1
    # shapely already uses the correct formulation for centroids
    polygon_rings["centroids"] = shapely.centroid(polygon_rings.geometry)
    # the inertia of parts applies to the overall center of mass:
    original_centroids = shapely.centroid(ga)
    polygon_rings["collection_centroid"] = original_centroids[collection_ix]
    # proportional to the squared distance between the original and part centroids:
    polygon_rings["radius"] = shapely.distance(
        polygon_rings.centroid.values, polygon_rings.collection_centroid.values
    )
    # now, we take the sum of (I+Ar^2) for each ring, treating the
    # contribution of holes as negative.
    # Then, we take the sum of all of the contributions
    return (
        polygon_rings.groupby(["collection_ix", "ring_within_geom_ix"])
        .apply(
            lambda ring_in_part: (
                (ring_in_part.moa + ring_in_part.radius**2 * ring_in_part.area)
                * ring_in_part.sign
            ).sum()
        ,include_groups=False)
        .groupby(level="collection_ix")
        .sum()
        .values
    )

def resultant_angle(gdf,vector_column='normal_vector',id_column='geom_id'):
    gdf_copy = gdf.copy()
    resultant_vector = gdf_copy[[vector_column,id_column,'geometry']].copy()
    resultant_vector = resultant_vector.groupby(id_column).agg({vector_column:'sum'}).reset_index()
    resultant_vector = resultant_vector.rename(columns={vector_column:'resultant_vector'})
    gdf_copy = gdf_copy.merge(resultant_vector,on=id_column,how='left')

    gdf_copy['angle'] = gdf_copy.apply(lambda x: pd.Series(get_angle_90(x[vector_column],x['resultant_vector'],x[id_column],x[id_column])),axis=1)
    return gdf_copy


def explode_exterior_and_interior_rings(gdf):
    gdf_copy = gdf.copy()
    crs = gdf.crs
    gdf_copy['exterior'] = gdf_copy.exterior
    gdf_copy['interiors'] = gdf_copy.interiors
    gdf_copy.geometry = gdf_copy.apply(lambda x: shapely.MultiLineString(x['interiors'] + [x['exterior']]),axis=1)
    gdf_copy.crs = crs
    gdf_copy = gdf_copy.drop(columns=['exterior','interiors'])
    gdf_copy = gdf_copy.explode().reset_index(drop=True)
    return gdf_copy

def eq_circle_intertia(area):
    r = np.sqrt(area / np.pi)
    return 0.25*np.pi*r**4

def circunscribed_square(geoms:gpd.GeoDataFrame|gpd.GeoSeries,dir_1_x,dir_1_y,dir_2_x,dir_2_y,return_length:bool=False):
    percentile = 0
    if percentile > 0:
        geometry = geoms.geometry.exterior
    else:
        geometry = geoms.geometry.convex_hull.exterior

    df = pd.DataFrame(
        {
            'x':geometry.apply(lambda x:np.array(x.coords)[:,0]),
            'y':geometry.apply(lambda x:np.array(x.coords)[:,1]),
            'dir_1_x':dir_1_x,
            'dir_1_y':dir_1_y,
            'dir_2_x':dir_2_x,
            'dir_2_y':dir_2_y
        }
    )
    df['pdirs_x_coords'] = df.apply(lambda row: (row['x'] * row['dir_2_y'] + row['y'] * (-row['dir_2_x'])) * 1 / (row['dir_1_x']*row['dir_2_y'] - row['dir_1_y']*row['dir_2_x']), axis=1)
    df['pdirs_y_coords'] = df.apply(lambda row: (row['x'] * (-row['dir_1_y']) + row['y'] * row['dir_1_x']) * 1 / (row['dir_1_x']*row['dir_2_y'] - row['dir_1_y']*row['dir_2_x']), axis=1)
    
    if return_length:
        if percentile > 0:
            length_1 = df['pdirs_x_coords'].apply(lambda arr: np.percentile(arr,percentile/100) - np.percentile(arr,1-percentile/100)).tolist()
            length_2 = df['pdirs_y_coords'].apply(lambda arr: np.percentile(arr,percentile/100) - np.percentile(arr,1-percentile/100)).tolist()
        else:
            length_1 = df['pdirs_x_coords'].apply(lambda arr: arr.max() - arr.min()).tolist()
            length_2 = df['pdirs_y_coords'].apply(lambda arr: arr.max() - arr.min()).tolist()
        
        return length_1, length_2
    else:
        if percentile > 0:
            df['min_pdir_x'] = df['pdirs_x_coords'].apply(lambda x: np.percentile(x,percentile/100))
            df['max_pdir_x'] = df['pdirs_x_coords'].apply(lambda x: np.percentile(x,1-percentile/100))
            df['min_pdir_y'] = df['pdirs_y_coords'].apply(lambda x: np.percentile(x,percentile/100))
            df['max_pdir_y'] = df['pdirs_y_coords'].apply(lambda x: np.percentile(x,1-percentile/100))
        else:
            df['min_pdir_x'] = df['pdirs_x_coords'].apply(np.min)
            df['max_pdir_x'] = df['pdirs_x_coords'].apply(np.max)
            df['min_pdir_y'] = df['pdirs_y_coords'].apply(np.min)
            df['max_pdir_y'] = df['pdirs_y_coords'].apply(np.max)

        df['square_1_x'] = df.apply(lambda row: (row['min_pdir_x'] * row['dir_1_x'] + row['min_pdir_y'] * row['dir_2_x']), axis=1)
        df['square_1_y'] = df.apply(lambda row: (row['min_pdir_x'] * row['dir_1_y'] + row['min_pdir_y'] * row['dir_2_y']), axis=1)
        df['square_2_x'] = df.apply(lambda row: (row['max_pdir_x'] * row['dir_1_x'] + row['min_pdir_y'] * row['dir_2_x']), axis=1)
        df['square_2_y'] = df.apply(lambda row: (row['max_pdir_x'] * row['dir_1_y'] + row['min_pdir_y'] * row['dir_2_y']), axis=1)
        df['square_3_x'] = df.apply(lambda row: (row['max_pdir_x'] * row['dir_1_x'] + row['max_pdir_y'] * row['dir_2_x']), axis=1)
        df['square_3_y'] = df.apply(lambda row: (row['max_pdir_x'] * row['dir_1_y'] + row['max_pdir_y'] * row['dir_2_y']), axis=1)
        df['square_4_x'] = df.apply(lambda row: (row['min_pdir_x'] * row['dir_1_x'] + row['max_pdir_y'] * row['dir_2_x']), axis=1)
        df['square_4_y'] = df.apply(lambda row: (row['min_pdir_x'] * row['dir_1_y'] + row['max_pdir_y'] * row['dir_2_y']), axis=1)

        df['square'] = df.apply(lambda row: shapely.Polygon([
            [row['square_1_x'],row['square_1_y']],
            [row['square_2_x'],row['square_2_y']],
            [row['square_3_x'],row['square_3_y']],
            [row['square_4_x'],row['square_4_y']],
            [row['square_1_x'],row['square_1_y']]
        ]), axis=1)
        return gps.GeoSeries(list(df['square']),crs=geoms.crs)
