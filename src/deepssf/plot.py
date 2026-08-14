"""Interactive (web-map) visualisation of observed and simulated trajectories.

Static matplotlib figures are built inline in the example notebooks; this module
holds the one plot that needs a third-party library and a coordinate transform,
so it is worth keeping in the package.

``folium`` is an optional dependency — install it with ``pip install folium``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

# Named basemaps.  Each entry is (tile URL template, attribution string).
BASEMAPS: dict[str, tuple[str, str]] = {
    "esri_imagery": (
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "Esri, Maxar, Earthstar Geographics",
    ),
    "esri_topo": (
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
        "Esri",
    ),
    "openstreetmap": ("OpenStreetMap", "OpenStreetMap contributors"),
}

# Qualitative palette, cycled when several trajectories are drawn.
TRAJECTORY_COLOURS: tuple[str, ...] = (
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
)


def _to_lonlat(
    x: Iterable[float], y: Iterable[float], src_crs: str
) -> tuple[np.ndarray, np.ndarray]:
    """Reproject projected coordinates to WGS84 longitude/latitude.

    Uses ``rasterio.warp.transform`` rather than pyproj: rasterio is already a
    dependency, and its transform is unambiguously (x, y) ordered, avoiding the
    lat/lon axis-order trap of ``pyproj.Transformer`` without ``always_xy=True``.
    """
    from rasterio.warp import transform as warp_transform

    xs = np.asarray(list(x), dtype=float)
    ys = np.asarray(list(y), dtype=float)
    lon, lat = warp_transform(src_crs, "EPSG:4326", xs, ys)
    return np.asarray(lon), np.asarray(lat)


def _coord_pairs(
    df: pd.DataFrame, x_col: str, y_col: str, src_crs: str
) -> list[list[float]]:
    """Return ``[[lat, lon], ...]`` pairs — the order folium expects."""
    sub = df[[x_col, y_col]].dropna()
    lon, lat = _to_lonlat(sub[x_col], sub[y_col], src_crs)
    return [[float(a), float(b)] for a, b in zip(lat, lon, strict=True)]


def plot_trajectories_folium(
    trajectories: pd.DataFrame | Sequence[pd.DataFrame] | None = None,
    observed: pd.DataFrame | None = None,
    src_crs: str = "EPSG:3112",
    x_col: str = "x",
    y_col: str = "y",
    id_col: str = "trajectory_id",
    observed_x_col: str = "x1_",
    observed_y_col: str = "y1_",
    observed_id_col: str | None = None,
    basemap: str = "esri_imagery",
    zoom_start: int = 11,
    centre: tuple[float, float] | None = None,
    colours: Sequence[str] = TRAJECTORY_COLOURS,
    observed_colour: str = "#ffffff",
    weight: float = 2.0,
    opacity: float = 0.85,
    observed_as_points: bool = True,
    max_observed_points: int = 5000,
    markers: bool = True,
    save_path: str | None = None,
):
    """Draw simulated (and optionally observed) trajectories on a web map.

    Parameters
    ----------
    trajectories:
        Simulated locations.  Either a single DataFrame — split into separate
        lines on *id_col* when that column is present — or a sequence of
        DataFrames, one per trajectory.
    observed:
        Observed GPS locations to overlay, e.g. ``step_df``.  Drawn as points
        by default because a step dataframe usually holds several individuals
        and a single line would connect them with spurious jumps.
    src_crs:
        CRS of the input coordinates (the projected CRS of the rasters and GPS
        data).  Everything is reprojected to EPSG:4326 for the map.
    x_col, y_col, id_col:
        Column names in *trajectories*.
    observed_x_col, observed_y_col, observed_id_col:
        Column names in *observed*.  When *observed_id_col* is given and
        *observed_as_points* is False, one line is drawn per individual.
    basemap:
        Key of :data:`BASEMAPS` ("esri_imagery", "esri_topo", "openstreetmap")
        or a raw XYZ tile-URL template.
    zoom_start, centre:
        Initial view.  *centre* is ``(lat, lon)``; by default the map is
        centred on the mean of the plotted simulated (or observed) locations.
    colours:
        Palette cycled across trajectories.
    observed_as_points:
        Draw *observed* as circle markers (True) or as connected lines (False).
    max_observed_points:
        Observed locations are thinned to at most this many points; folium
        writes every marker into the page as its own JavaScript object, so this
        is what keeps the saved HTML to a few MB rather than tens of MB.
    markers:
        Add start (green) and end (red) markers to each simulated trajectory.
    save_path:
        If given, also write the map to a standalone HTML file.

    Returns
    -------
    folium.Map
        Display it by leaving it as the last expression in a notebook cell.
    """
    try:
        import folium
    except ImportError as exc:  # pragma: no cover - trivial guard
        raise ImportError(
            "plot_trajectories_folium requires folium: pip install folium"
        ) from exc

    # --- Normalise the trajectory input to a list of (label, DataFrame) ---
    traj_list: list[tuple[str, pd.DataFrame]] = []
    if trajectories is not None:
        if isinstance(trajectories, pd.DataFrame):
            if id_col in trajectories.columns:
                traj_list = [
                    (f"simulated {tid}", sub)
                    for tid, sub in trajectories.groupby(id_col, sort=True)
                ]
            else:
                traj_list = [("simulated", trajectories)]
        else:
            traj_list = [
                (f"simulated {i}", df) for i, df in enumerate(trajectories)
            ]

    if not traj_list and observed is None:
        raise ValueError("Nothing to plot: pass trajectories, observed, or both.")

    # --- Basemap tiles ---
    if basemap in BASEMAPS:
        tiles, attr = BASEMAPS[basemap]
    else:
        tiles, attr = basemap, "tiles"

    # --- Reproject everything up front (one warp call per layer) ---
    traj_coords = [
        (label, _coord_pairs(df, x_col, y_col, src_crs)) for label, df in traj_list
    ]

    obs_frames: list[tuple[str, pd.DataFrame]] = []
    if observed is not None:
        if observed_id_col is not None and observed_id_col in observed.columns:
            obs_frames = [
                (str(oid), sub)
                for oid, sub in observed.groupby(observed_id_col, sort=True)
            ]
        else:
            obs_frames = [("observed", observed)]

    # --- Map centre ---
    if centre is None:
        if traj_coords and traj_coords[0][1]:
            all_pts = [p for _, pts in traj_coords for p in pts]
        else:
            all_pts = _coord_pairs(
                obs_frames[0][1], observed_x_col, observed_y_col, src_crs
            )
        arr = np.asarray(all_pts, dtype=float)
        centre = (float(arr[:, 0].mean()), float(arr[:, 1].mean()))

    fmap = folium.Map(location=list(centre), tiles=tiles, attr=attr,
                      zoom_start=zoom_start, control_scale=True)

    # --- Observed layer ---
    if obs_frames:
        obs_group = folium.FeatureGroup(name="observed", show=True)
        for label, df in obs_frames:
            pts = _coord_pairs(df, observed_x_col, observed_y_col, src_crs)
            if observed_as_points:
                # Thin by a fixed stride so the thinned track still spans the
                # full extent (a head slice would show only the first days).
                stride = max(1, len(pts) // max_observed_points)
                for lat, lon in pts[::stride]:
                    folium.CircleMarker(
                        location=[lat, lon], radius=1.2, color=observed_colour,
                        fill=True, fill_opacity=0.5, opacity=0.5, weight=0.5,
                    ).add_to(obs_group)
            else:
                folium.PolyLine(
                    locations=pts, color=observed_colour, weight=weight,
                    opacity=opacity, tooltip=f"observed {label}",
                ).add_to(obs_group)
        obs_group.add_to(fmap)

    # --- Simulated layers (one toggleable group per trajectory) ---
    for i, (label, pts) in enumerate(traj_coords):
        if not pts:
            continue
        colour = colours[i % len(colours)]
        group = folium.FeatureGroup(name=label, show=True)
        folium.PolyLine(
            locations=pts, color=colour, weight=weight, opacity=opacity,
            tooltip=label,
        ).add_to(group)
        if markers:
            folium.CircleMarker(
                location=pts[0], radius=5, color="#00ff00", fill=True,
                fill_opacity=1.0, tooltip=f"{label} start",
            ).add_to(group)
            folium.CircleMarker(
                location=pts[-1], radius=5, color="#ff0000", fill=True,
                fill_opacity=1.0, tooltip=f"{label} end",
            ).add_to(group)
        group.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)

    if save_path is not None:
        fmap.save(save_path)
        print(f"Map saved: {save_path}")

    return fmap
