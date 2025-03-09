# Footprint Attributes

**Footprint Attributes** is a Python package built using GeoPandas to calculate various geometric indices related to building footprint geometry and seismic risk.

<div align="center">
  <img src="examples/example_img.png" alt="screenshot" width="500"/>
</div>

---

## Installation

To install the package, use the following command:

```bash
pip install git+https://github.com/GeomaticsCaminosUPM/footprint_attributes.git
```

---

## Features

### 1. **Relative Position of Buildings**
This feature determines if a building touches other structures (relative position within the city block). It calculates "forces" that neighboring structures exert on the building, proportional to the contact area (length of touching footprints multiplied by building height) in the normal direction of the touching plane.

#### **Function: `get_forces_gdf`**
```python
get_forces_gdf(
    geoms: gpd.GeoDataFrame,
    buffer: float = 0,
    height_column: str = None,
    min_radius: float = 0
) -> gpd.GeoDataFrame
```

##### **Parameters:**
- **`geoms`** (`gpd.GeoDataFrame`):  
  A GeoDataFrame containing building footprints as polygon geometries.
  
- **`buffer`** (`float`):  
  Buffer distance in meters to determine if two buildings are considered touching.

- **`height_column`** (`str`, optional):  
  The column name in `geoms` specifying the building height in meters.  
  If `None`, all buildings will have a uniform height of `1`.

- **`min_radius`** (`float`, optional):  
  Minimum distance multiplier used to exclude forces that would otherwise increase momentum.  
  Forces with a distance below a threshold (`min_radius * equivalent radius`) will only contribute to momentum calculation if they decrease the momentum.  
  The equivalent radius of a building is the radius of a circle with the same area as the building's footprint.

---

##### **Output:**
Returns a `gpd.GeoDataFrame` with the following columns:

- **`height`**:  
  The building height. If `height_column` is `None`, the height will default to `1`.

- **`angular_acc`**:  
  The angular acceleration, calculated as:

$$\text{angular acc} = \frac{\text{momentum} \cdot \text{area}}{\text{inertia}}$$
   
  Where **momentum** is calculated as:
  
$$\text{momentum} = \sum \(\text{distance} \cdot |\text{force}_i|\)$$

- **`force`**:  
  The magnitude of the resultant force acting on the footprint, normalized by the square root of the area:
  
$$\text{force} = \left| \sum \text{force}_i \right|$$

- **`confinement_ratio`**:  
  The proportion of total forces that are confined (counterbalanced by opposing forces):
    
$$\text{confinement ratio} = \frac{\sum |\text{force}_i| - \left| \sum \text{force}_i \right|}{\left| \sum \text{force}_i \right|}$$

- **`angle`**:  
  The normalized sum of the angles between individual forces and the resultant force:
   
$$\text{angle} = \frac{\sum \left( |\text{force}_i| \cdot \text{angle}(\text{force}_i, \sum \text{force}_j) \right)}{\left| \sum \text{force}_i \right|}$$

- **`geometry`**:  
  The original building footprint geometries (from the input GeoDataFrame).

**Note:** The row indices in the output GeoDataFrame will be the same as the indices in the input `geoms` GeoDataFrame.

---

#### **Function: `relative_position`**
```python
relative_position(
    forces: gpd.GeoDataFrame,
    min_angular_acc: float = 0.0825,
    min_confinement: float = 1,
    min_angle: float = 0.78,
    min_force: float = 0.166
) -> list
```

##### Parameters:
- **`forces`** (`gpd.GeoDataFrame`): Output GeoDataFrame from `get_forces_gdf`.
  
- **`min_force`** (`float`, optional): Minimum force threshold (default: `0.166`; e.g., for a square building with height 1 and side length 1, if a touching structure covers only 1/6 of one side, the resultant force would be 1/6.)
  
- **`min_angle`** (`float`, optional): Minimum angle threshold (default: `π/4` radians or 45 degrees; e.g., for a square building with height 1 and side length 1, if a touching structure covers only 1/6 of one side, the resultant force would be 1/6.)
  
- **`min_confinement`** (`float`, optional): Minimum confinement threshold (default: `1`; equal amounts of confined and resultant forces).
  
- **`min_angular_acc`** (`float`, optional): Minimum angular acceleration threshold \frac{momentum * area}{inertia}$ (default: `2.133`; e.g., for a rectangular building with height 1 and sides of length 1 and 0.5, 
                            a touching structure covering 1/3 of two sides in the worst case would have an anuglar acceleration of 2.133)

##### Output:
Returns a list of relative positions for buildings, classified as:
1. **"torque"**: High angular acceleration and class **confined** or **corner**.
2. **"confined"**: Touches on both lateral sides.
3. **"corner"**: Touches at a corner.
4. **"lateral"**: Touches on one side.
5. **"isolated"**: No touching structures.

**Note:** The order of the list corresponds to the input `forces` rows.

---

### 2. **Shape Irregularity**
Measures geometric irregularity of building footprints using various indices.

#### **Polsby-Popper Index**
Measures shape compactness (similarity to a circle).
##### **Formula:**
    
  $$\text{Polsby-Popper Index} = \frac{4 \pi A}{P^2}$$
  
  where:
  - \( A \): Area of the polygon.
  - \( P \): Perimeter of the polygon.

##### **Function: `polsby_popper`**
```python
polsby_popper(geoms: gpd.GeoDataFrame, convex_hull: bool = False) -> list
```

##### **Parameters:** 
- **`geoms`** (`gpd.GeoDataFrame`): GeoDataFrame with building footprint geometries.
- **`convex_hull`** (`bool`, optional): Use the convex hull of the geometries instead to compute the polsby popper index (default: `False`).
##### **Output:**
List of `polsby_popper` values corresponding to `geoms` rows.

---

#### **Custom Irregularity Index**
Quantifies the irregularity of footprints based on the diference between the boundary of the footprint and the convex hull.
##### **Formula:**
  
  $$\text{Custom Irregularity Index} = \frac{l \cdot d}{L}$$
  
  where:
  - \( l \): Length of the geometries outside the convex hull.
  - \( d \): Distance of the center of gravity of the geometries outside the hull to the convex hull.
  - \( L \): Total convex hull length.

**Note:** Footprint polygons and convex hulls are transformed into `LineStrings` based on their boundary.

##### **Function: `shape_irregularity`**
```python
shape_irregularity(geoms: gpd.GeoDataFrame) -> list
```

##### **Parameters**:  
- **`geoms`** (`gpd.GeoDataFrame`): GeoDataFrame with building footprint geometries.
##### **Output**: 
List of `shape_irregularity` values corresponding to `geoms` rows.

---

#### **Inertia Irregularity**
Compares the inertia of a polygon to a circle with the same area.
##### **Formula**:
  
  $$\text{Inertia Irregularity} = \frac{\text{Inertia of Equivalent Circle}}{\text{Inertia of Polygon}}$$

##### Function: `inertia_irregularity`
```python
inertia_irregularity(geoms: gpd.GeoDataFrame) -> list
```

##### **Parameters**:  
- **`geoms`** (`gpd.GeoDataFrame`): GeoDataFrame with building footprint geometries.
##### **Output**: 
List of `inertia_irregularity` values corresponding to `geoms` rows.

---



