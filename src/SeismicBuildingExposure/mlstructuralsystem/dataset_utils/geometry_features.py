"""
Geometry-derived structural feature engineering.

Functions in this module compute seismic-engineering shape indicators
(eccentricity, setbacks, slenderness, contact forces, relative position)
directly from building footprint geometries, using the `footprint` package.
"""

from ... import footprint


def add_position_features(gdf, cfg):
    if ("relative_position" not in cfg.FEATURES) or ("relative_position" in gdf.columns):
        return gdf

    gdf = gdf.copy()

    if any(gdf.geometry.type == 'MultiPolygon'):
        print("\n" + "="*60)
        print("⚠️ WARNING: There are multiplart geometries. Exploding geometries.")
        print("="*60 + "\n")
        gdf = gdf.explode().reset_index(drop=True)

    mask = gdf.geometry.type.str.contains("Polygon", na=False)

    n_removed = (~mask).sum()

    if n_removed > 0:
        print("\n" + "="*60)
        print(f"⚠️ WARNING: Removed {n_removed} rows with non-Polygon or invalid geometries")
        print("="*60 + "\n")

        gdf = gdf[mask]
        
    if "id" not in gdf.columns:
        gdf["id"] = gdf.index

    gdf = gdf.to_crs(gdf.estimate_utm_crs())

    if "height" in gdf.columns:
        height_column = "height"
    else:
        height_column = None

    forces = footprint.position.contact_forces_df(
        gdf.copy(),
        height_column=height_column,
        buffer=cfg.POSITION_BUFFER,
        min_radius=cfg.POSITION_MIN_RADIUS
    )

    if any(field.startswith("contact_") for field in cfg.FEATURES):
        gdf["contact_force"] = list(forces["force"])
        gdf["contact_confinement_ratio"] = list(forces["confinement_ratio"])
        gdf["contact_angular_acc"] = list(forces["angular_acc"])
        gdf["contact_angle"] = list(forces["angle"])

    gdf['relative_position'] = footprint.position.relative_position(
        forces,
        min_angular_acc=cfg.POSITION_MIN_ANGULAR_ACC,
        min_confinement=cfg.POSITION_MIN_CONFINEMENT,
        min_angle=cfg.POSITION_MIN_ANGLE,
        min_force=cfg.POSITION_MIN_FORCE
    )

    gdf = gdf.to_crs(4326)
    return gdf 


def add_irregularity_features(gdf, cfg):
    def _needs_prefix(features, cfg, prefix, do_fsi=False):
        """
        Returns True if:
        - any requested field starts with `prefix`, OR
        - FSI dependency columns reference that prefix
        """
        if any(f.startswith(prefix) for f in features):
            return True

        if not do_fsi:
            return False

        ecc = getattr(cfg, "FSI_ECCENTRICITY_COL", "")
        setb = getattr(cfg, "FSI_SETBACK_COL", "")
        slen = getattr(cfg, "FSI_SLENDERNESS_COL", "")

        return (
            ecc.startswith(prefix)
            or setb.startswith(prefix)
            or slen.startswith(prefix)
        )

    gdf = gdf.copy()

    # -------------------------
    # Geometry checks
    # -------------------------
    if any(gdf.geometry.type == 'MultiPolygon'):
        print("\n" + "="*60)
        print("⚠️ WARNING: There are multiplart geometries. Exploding geometries.")
        print("="*60 + "\n")
        gdf = gdf.explode().reset_index(drop=True)

    mask = gdf.geometry.type.str.contains("Polygon", na=False)

    n_removed = (~mask).sum()

    if n_removed > 0:
        print("\n" + "="*60)
        print(f"⚠️ WARNING: Removed {n_removed} rows with non-Polygon or invalid geometries")
        print("="*60 + "\n")

        gdf = gdf[mask]

    # -------------------------
    # ID column
    # -------------------------
    if "id" not in gdf.columns:
        gdf["id"] = gdf.index

    # -------------------------
    # Normalize geometry
    # -------------------------
    gdf = gdf.explode(ignore_index=True)

    try:
        gdf = gdf.to_crs(gdf.estimate_utm_crs())
    except Exception:
        pass

    height_col = "height" if "height" in gdf.columns else None
    features = set(cfg.FEATURES)

    # -------------------------
    # FSI dependency flag
    # -------------------------
    do_fsi = "fsi" in features

    # -------------------------
    # Feature requirements
    # -------------------------
    needs_ec8 = _needs_prefix(features, cfg, "EC8_", do_fsi)
    needs_cr = _needs_prefix(features, cfg, "CR_", do_fsi)
    needs_ntc = _needs_prefix(features, cfg, "NTC_", do_fsi)
    needs_asce7 = _needs_prefix(features, cfg, "ASCE7_", do_fsi)
    needs_gndt = _needs_prefix(features, cfg, "GNDT_", do_fsi)

    # -------------------------
    # EC8
    # -------------------------
    if needs_ec8:
        ec8 = footprint.shape.eurocode_8_df(gdf)
        gdf["EC8_eccentricity_ratio"] = list(ec8["eccentricity_ratio"])
        gdf["EC8_radius_ratio"] = list(ec8["radius_ratio"])
        gdf["EC8_compactness"] = list(ec8["compactness"])
        gdf["EC8_direction_eccentricity"] = list(ec8["angle_eccentricity"])

    # -------------------------
    # Costa Rica
    # -------------------------
    if needs_cr:
        cr = footprint.shape.codigo_sismico_costa_rica_df(gdf)
        gdf["CR_eccentricity_ratio"] = list(cr["eccentricity_ratio"])
        gdf["CR_direction_eccentricity"] = list(cr["angle"])

    # -------------------------
    # NTC Mexico
    # -------------------------
    if needs_ntc:
        mx = footprint.shape.NTC_mexico_df(gdf)
        gdf["NTC_setback_ratio"] = list(mx["setback_ratio"])
        gdf["NTC_hole_ratio"] = list(mx["hole_ratio"])

    # -------------------------
    # ASCE 7
    # -------------------------
    if needs_asce7:
        asce = footprint.shape.asce_7_df(gdf)
        gdf["ASCE7_setback_ratio"] = list(asce["setback_ratio"])
        gdf["ASCE7_hole_ratio"] = list(asce["hole_ratio"])
        gdf["ASCE7_parallelity_angle"] = list(asce["parallelity_angle"])

    # -------------------------
    # GNDT Italy
    # -------------------------
    if needs_gndt:
        gndt = footprint.shape.gndt_italy_df(
            gdf,
            min_length=cfg.GNDT_MIN_LENGTH,
            min_area=cfg.GNDT_MIN_AREA
        )
        gdf["GNDT_main_shape_slenderness"] = list(gndt["beta_1_main_shape_slenderness"])
        gdf["GNDT_setback_ratio"] = list(gndt["beta_2_setback_ratio"])
        gdf["GNDT_eccentricity_ratio"] = list(gndt["beta_4_eccentricity_ratio"])
        gdf["GNDT_setback_slenderness"] = list(gndt["beta_6_setback_slenderness"])

    # -------------------------
    # Slenderness metrics
    # -------------------------
    if ("slenderness_elevation" in features) or ("slenderness_elevation" == cfg.FSI_SLENDERNESS_COL):
        if height_col is None:
            raise ValueError("Column 'height' is needed")

        a = footprint.shape.get_a(gdf)
        gdf["slenderness_elevation"] = gdf[height_col] / a

    if ("slenderness_inertia" in features) or ("slenderness_inertia" == cfg.FSI_SLENDERNESS_COL):
        val, ang = footprint.shape.inertia_slenderness(gdf, return_direction=True)
        gdf["slenderness_inertia"] = list(val)
        gdf["inertia_direction"] = list(ang)

    if ("slenderness_bbox" in features) or ("slenderness_bbox" == cfg.FSI_SLENDERNESS_COL):
        val, ang = footprint.shape.min_bbox_slenderness(gdf, return_direction=True)
        gdf["slenderness_bbox"] = list(val)
        gdf["bbox_direction"] = list(ang)

    if ("slenderness_circunscribed" in features) or ("slenderness_circunscribed" == cfg.FSI_SLENDERNESS_COL):
        val, ang = footprint.shape.circunscribed_slenderness(gdf, return_direction=True)
        gdf["slenderness_circunscribed"] = list(val)
        gdf["circunscribed_direction"] = list(ang)

    if ("inertia_vs_circle" in features) or ("inertia_vs_circle" == cfg.FSI_SLENDERNESS_COL):
        gdf["inertia_vs_circle"] = footprint.shape.inertia_circle(gdf)

    # -------------------------
    # FSI classification
    # -------------------------
    if do_fsi:
        gdf["fsi"] = "regular"

        gdf.loc[gdf[cfg.FSI_ECCENTRICITY_COL] > cfg.FSI_ECCENTRICITY_VAL, "fsi"] = "eccentricity"
        gdf.loc[gdf[cfg.FSI_SETBACK_COL] > cfg.FSI_SETBACK_VAL, "fsi"] = "setbacks"
        gdf.loc[gdf[cfg.FSI_SLENDERNESS_COL] > cfg.FSI_SLENDERNESS_VAL, "fsi"] = "slenderness"

        if "regularity_boolean" in features:
            gdf["regularity_boolean"] = gdf["fsi"].eq("regular")

    return gdf