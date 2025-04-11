Under development

# SeismicBuildingExposure

**SeismicBuildingExposure** is a Python package that computes everything you need to estimate the seismic exposure of buildings using only geospatial data and no field surveys.

---

## Installation

To install the package, use the following command:

```bash
pip install git+https://github.com/GeomaticsCaminosUPM/SeismicBuildingExposure.git
```

---

## Modules
  
### 1. MLfootprint 
Machine learning AI model for automatic instance segmentation of building footprints. 

- Fine tune the SAM2 model for assisted segmentation.
- Fine tune the maskformer model for automatic segmentation.
- Run new images (interference) on the SAM2 model for assisted segmentation.
- Run new images (interference) on the maskformer model for automatic segmentation.

To create and download a dataset from publically available sources use the **data** module.

### 2. MLstructural_system 

Predicts the structural system of a building using the data extracted from the **footprint**, **height** and **remote_sensing** modules.

- Fine tune or train your own bayesian model with your own survey.
- Predict using one of our pre-trained model on your own dataset.

### 3. data 

The **data** module is based on the **GeoVisionDataset** library and provides functions adapted to create building footprint datasets using publically available datasources.

### 4. footprint 

This module is divided in 2:

- **position**: Relative position of the building (in a row, on a corner, isolated, etc.)
- **irregularity**: Footprint irregularity according to international building codes.

### 5. height 

Provides functions to process a .ply point cloud and get building heights and other height related data. 

- **DSM**: Digital surface model (raster image) form a .ply point cloud.
- **DTM**: Digital terrain model (raster image) using the DSM as input and considering ground points to be on streets downloaded from OpenStreetMap.
- **height**: Building height, roof steepness, ground altitude and ground steepness  on every building footprint.
- **irregularity**: Irregularity in elevation according to international building codes.

### 6. remote_sensing 

- **photogrammetry**: COLMAP open source photogrammetry workflow to get a .ply dense point cloud from drone images.
- **roof_material**: Estimation of the roof material using LANDSAT and SENTINEL-2 spectral information.
- **year**: Estimation of the year of first construction and year of last building change using LANDSAT sattelite imagery. 
