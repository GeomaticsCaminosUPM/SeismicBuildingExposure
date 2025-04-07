# SeismicBuildingExposure.footprint module

The **footprint** module focuses on tools to characterize different structural behaviour modifiers. For tools to help with footprint geometry aquisition please go to **data.footprints** or **MLfootprints**. 
---

## Installation

To install the package, use the following command:

```bash
pip install git+https://github.com/GeomaticsCaminosUPM/SeismicBuildingExposure.git@main#subdirectory=SeismicBuildingExposure/SeismicBuildingExposure/footprint
```

---

## Features

### 1. **Position module** (Relative position of buildings)
This feature determines if a building touches other structures (relative position within the city block). It calculates "forces" that neighboring structures exert on the building, proportional to the contact area (length of touching footprints multiplied by building height) in the normal direction of the touching plane.

Contact forces are computed to help determine the realtive position class:
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

Relative position classes are:
1. **"torque"**: High angular acceleration and class **confined** or **corner**.
2. **"confined"**: Touches on both lateral sides.
3. **"corner"**: Touches at a corner.
4. **"lateral"**: Touches on one side.
5. **"isolated"**: No touching structures.

<div align="center">
  <img src="https://github.com/GeomaticsCaminosUPM/SeismicBuildingExposure/blob/main/images/san_jose_relative_position.png" alt="screenshot" width="500"/>
</div>

---

### 2. **Irregularity module**
Measures geometric irregularity of **building footprints** using indices from different building codes around the world.

#### 2.1. Eurocode 8

Irregularity is measured following the [Eurocode 8 standard](https://www.phd.eng.br/wp-content/uploads/2015/02/en.1998.1.2004.pdf).

The calculated parameters are:

- **Excentricity Ratio**:  
  $\text{excentricity ratio} = \frac{\text{excentricity}}{\text{torsional radius}}$  
  This considers the worst-case value across all possible directions.

- **Radius Ratio**:  
  $\text{radius ratio} = \frac{\text{torsional radius}}{\text{radius of gyration}}$


- **Slenderness**:  
  Typically calculated as $\frac{L_1}{L_2}$, where $L_1$ and $L_2$ are the sides of the footprint.  
  Since polygonal shapes may not clearly resemble rectangles, we use:  
  $\sqrt{\frac{I_1}{I_2}}$  
  where $I_1$ and $I_2$ are the principal values of the inertia tensor.

- **Compactness**:  
  $\text{compactness} = \frac{\text{area of polygon (with all holes filled)}}{\text{area of convex hull}}$

The function `eurocode_8_df` returns the **weakest direction** as an angle (in degrees) with respect to the **north (in UTM coordinates)**.

##### Definitions:

- **Excentricity**: Distance between the center of mass (centroid of the polygon) and the center of stiffness (centroid of the boundary).
- **Torsional Radius**:  
  $\sqrt{\frac{I_t}{I_j}}$  
  where:
  - $I_t$: Inertia in the Z-direction through the center of stiffness.  
  - $I_j$: Inertia through the center of mass along the axis perpendicular to the calculation direction.

- **Radius of Gyration**:  
  $\sqrt{\frac{I_0}{\text{area}}}$  
  where:
  - $I_0$: Inertia in the Z-direction through the center of mass.

##### Parameter Limits:

| Parameter             | Limit        |
|----------------------|--------------|
| Excentricity Ratio    | < 0.3        |
| Radius Ratio          | < 1.0        |
| Slenderness           | < 4.0        |
| Compactness           | > 0.95       |

#### 2.2. Costa Rica Código Sísmico Norm

Irregularity is measured following the [Seismic Code of Costa Rica](https://www.codigosismico.or.cr/).

The calculated parameter is:

- **Excentricity Ratio**:  
  $\text{excentricity ratio} = \frac{\text{excentricity}}{\text{dimension}}$  
  considering the **weakest possible direction**.

The function `codigo_sismico_costa_rica_df` returns the **weakest direction** as an angle (in degrees) with respect to the **north (in UTM coordinates)**.

##### Definitions:

- **Excentricity**:  
  The distance between the **center of mass** (centroid of the polygon) and the **center of stiffness** (centroid of the boundary).

- **Dimension**:  
  The length of the shape in the considered direction. For rectangles, this is straightforward, but for an arbitrary polygon it is computed as:  
  $\text{dimension} = \sqrt{\text{area} \cdot \sqrt{\frac{I_i}{I_j}}}$  
  where:
  - $I_i$: Inertia in the considered direction (through the center of mass).
  - $I_j$: Inertia in the perpendicular direction (also through the center of mass).

##### Parameter Limits:

| Parameter            | Limit                        | Irregularity Level |
|---------------------|------------------------------|--------------------|
| Excentricity Ratio   | $< 0.05$                      | Regular             |
| Excentricity Ratio   | $0.05 < \frac{e}{d} < 0.25$   | Moderate            |
| Excentricity Ratio   | $> 0.25$                      | High                |


<div align="center">
  <img src="https://github.com/GeomaticsCaminosUPM/SeismicBuildingExposure/blob/main/images/guatemala_cr_norm.png" alt="screenshot" width="500"/>
</div>

#### 2.3. Mexico NTC norm

Irregularity is measured following the [Mexico NTC norm]().

The calculated parameters are:

- **Setback Ratio**:  
  $\text{setback ratio} = \frac{\text{setback length}}{\text{side length}}$  
  considering the **worst of the two directions** and the **worst of all setbacks**.

- **Hole Ratio**:  
  $\text{hole ratio} = \frac{\text{hole width}}{\text{side length}}$  
  considering the **worst of the two directions** and the **worst of all holes**.

### Definitions:

- **Side Length**:  
  The footprint is circumscribed in the **smallest possible rectangle**, with sides aligned to the **principal axes of the inertia tensor** of the footprint.  
  The *length of side* refers to the side of this rectangle along either of the principal directions.

  For the **hole ratio**, we consider the **principal axes of each hole shape**, and measure the side length as a line passing through the **center of mass of the hole** in each principal direction.

- **Setback Length**:  
  Setbacks are defined as the **polygons formed by the difference between the convex hull and the footprint (with holes filled)**.  
  These setback polygons are also circumscribed in a rectangle whose sides are aligned with the **principal directions of the inertia tensor of the footprint**.  
  The *setback length* is the side of this rectangle in one of the two principal directions.  
  In the **setback ratio**, both the *setback length* and the *side length* must be taken in the **same direction** (parallel).

- **Hole Width**:  
  Each hole is circumscribed in a rectangle, with sides aligned to the **principal directions of the hole’s inertia tensor**.  
  The *hole width* is the length of this rectangle in each principal direction.

- **Max Slenderness**: In the context of the **setback ratio**, very thin irregularities caused by concave angles close to 180º comming from imperfections in the footprint,  can lead to a circumscribed rectangle with a disproportionately large side, even though such features may not represent a true setback. **Max slenderness** is the maximum slenderness of **setback circunscribed rectangles**. 

##### Parameter Limits:

| Parameter        | Limit |
|------------------|--------|
| Setback Ratio     | $< 0.4$ |
| Hole Ratio        | $< 0.4$ |

<div align="center">
  <img src="https://github.com/GeomaticsCaminosUPM/SeismicBuildingExposure/blob/main/images/santo_domingo_ntc.png" alt="screenshot" width="500"/>
</div>

#### 2.4. Geometric indices

##### **Polsby-Popper Index**
Measures shape compactness (similarity to a circle).
###### **Formula:**
    
  $$\text{Polsby-Popper Index} = \frac{4 \pi A}{P^2}$$
  
  where:
  - \( A \): Area of the polygon.
  - \( P \): Perimeter of the polygon.


##### **Convex Hull Momentum Index**
Quantifies the irregularity of footprints based on the diference between the boundary of the footprint and the convex hull.
###### **Formula:**
  
  $$\text{Convex Hull Momentum} = \frac{l \cdot d}{L}$$
  
  where:
  - \( l \): Length of the geometries outside the convex hull.
  - \( d \): Distance of the center of gravity of the geometries outside the hull to the convex hull.
  - \( L \): Total convex hull length.

**Note:** Footprint polygons and convex hulls are transformed into `LineStrings` based on their boundary.


##### **Inertia Circle Irregularity**
Compares the inertia of a polygon to a circle with the same area.
###### **Formula**:
  
  $$\text{Inertia Irregularity} = \frac{\text{Inertia of Equivalent Circle}}{\text{Inertia of Polygon}}$$


#### **Inertia Slenderness** 
Meassures the slenderness of the footrpint based on the relation of the two principal inertia values. 

##### **Formula** 

$$\text{Inertia Slenderness} = \sqrt{\frac{I_1}{I_2}}$$ 

where:
- $I_1$ maximum principal inertia value.
- $I_2$ minimum principal inertia value.


##### **Circunscribed Slenderness** 
Meassures the slenderness of the footrpint based on the relation of the sides of the circunscribed rectangle in the principal direction (minimum possible rectangle). 

###### **Formula** 

$$\text{Circunscribed Slenderness} = \frac{L_1}{L_2}$$ 

##### **Eurocode 8**

All indices from the Eurocode 8 are available as independent functions:

- Excentricity EC8
- Radius Ratio
- Slenderness
- Compactness

##### **Costa Rica Código Sísmico Norm**

All indices from the Costa Rica Seismic Code are available as independent functions:

- Excentricity CR

##### **Mexico NTC Norm** 

All indices from the Mexico NTC norm are available as independent functions:

- Setback Ratio
- Hole Ratio

