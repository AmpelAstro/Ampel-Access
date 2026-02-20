#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File:                icecube_tools.py
# License:             BSD-3-Clause
# Author:              cristozilleruelo
# Date:                11.2.2026
# Last Modified Date:  19.2.2026
# Last Modified By:    Felix Fischer

"""
icecube_tools
=============

Utilities for working with IceCube HEALPix probability maps (multi-order fits.gz):

- Plot a full-sky + zoom view (Mollweide + inset) of an alert skymap
- Plot a zoom-only view
- Select pixels covering a given containment probability
- Check if a (ra, dec) source lies inside the selected containment region

Notes
-----
These maps should be read via `ligo.skymap.io.fits.read_sky_map`.
"""

from __future__ import annotations

from typing import Any, Optional, Literal

import json
import os
from pathlib import Path
from collections.abc import Iterable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from itertools import cycle

import requests
from urllib.parse import urljoin
from html.parser import HTMLParser
import healpy as hp
import astropy_healpix as ah

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import QTable
from astropy.time import Time

import ligo.skymap.plot
from ligo.skymap.io.fits import read_sky_map

import mhealpy as mhp



_P = {
    "map_dir": "./cache_ampelaccess/IceCubeAlerts",
    "save_dir": "./cache_ampelaccess/IceCubeAlerts",
    "file_dir": "./cache_ampelaccess/IceCubeAlerts",
    "json_alerts": "./cache_ampelaccess/IceCubeAlerts",
}

# ------------------------------------------------------------------
# Retrieving and formating IceCube alerts
# ------------------------------------------------------------------

def _is_url(s: str) -> bool:
    """
    Check if a given string is a URL (starts with "http://" or "https://").
    """
    return str(s).startswith("http://") or str(s).startswith("https://")


def _download_to(
    map_url: str,
    map_dir: str | Path,
    *,
    timeout: float = 30.0,
    overwrite: bool = False,
) -> str:
    """
    Download a healpix map from a given URL and save it to a given directory.
    """
    file_name = map_url.split("/")[-1]
    map_dir = str(map_dir)
    os.makedirs(map_dir, exist_ok=True)
    map_path = os.path.join(map_dir, file_name)

    if os.path.exists(map_path) and not overwrite:
        return map_path

    r = requests.get(map_url, timeout=timeout)
    r.raise_for_status()

    with open(map_path, "wb") as f:
        f.write(r.content)

    return map_path


class _HrefParser(HTMLParser):
    """ 
    Check for existing links in an HTML document.
    Necessary to list all existing IceCube ROC skymaps.
    """
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for k, v in attrs:
            if k.lower() == "href" and v:
                self.hrefs.append(v)


def list_icecube_public_skymaps(
    index_url: str = "https://roc-2.icecube.wisc.edu/public/alerts/",
    *,
    timeout: float = 30.0,
) -> list[str]:
    """
    List absolute URLs of public IceCube ROC skymaps.
    This returns only what is present in the given directory.
    """
    r = requests.get(index_url, timeout=timeout)
    r.raise_for_status()

    p = _HrefParser()
    p.feed(r.text)

    urls: list[str] = []
    for href in p.hrefs:
        if href.endswith(".fits.gz") and "skymap" in href:
            urls.append(urljoin(index_url, href))

    # de-duplicate, stable order
    seen: set[str] = set()
    out: list[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)

    return out


def iter_icecube_public_alerts(
    index_url: str = "https://roc-2.icecube.wisc.edu/public/alerts/",
    *,
    download: bool = True,
    map_dir: str | Path = _P["map_dir"],
    timeout: float = 30.0,
    overwrite: bool = False,
    infer_alert_datetime: bool = False,
) -> list[dict[str, Any]]:
    """
    Iterate over all public IceCube ROC skymap URLs and cache if wished.
    """
    urls = list_icecube_public_skymaps(index_url=index_url, timeout=timeout)

    out: list[dict[str, Any]] = []
    for u in urls:
        fname = u.split("/")[-1]
        event_name = fname.split("_")[0] if "_" in fname else fname

        if download:
            map_path = _download_to(u, map_dir, timeout=timeout, overwrite=overwrite)
            d: dict[str, Any] = {"healpix_url": map_path, "event_name": event_name}

            if infer_alert_datetime:
                try:
                    _hpx, header = _read_skymap_fix_coordsys(map_path, nest=True)
                    d["alert_datetime"] = _infer_alert_datetime_from_header(header)
                except Exception:
                    d["alert_datetime"] = None

            out.append(d)
        else:
            out.append({"healpix_url": u, "event_name": event_name})

    return out
def convert_txt_alerts_to_fits(
    directory: str | Path,
    *,
    patterns: tuple[str, ...] = ("*.txt",),
    timeout: float = 30.0,
    overwrite: bool = False,
    copy_local_files: bool = False,
) -> dict[str, Any]:
    """
    Convert IceCube alert txt/json files inside `directory` into local multiorder FITS files (*.fits.gz).

    The txt file can be:
    - JSON list: [{...}, {...}]
    - NDJSON: one JSON dict per line

    Each alert dict must contain `healpix_url` (URL or local path).
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(str(directory))
    if not directory.is_dir():
        raise NotADirectoryError(str(directory))

    def _parse_alert_file(p: Path) -> list[dict[str, Any]]:
        raw = p.read_text(encoding="utf-8").strip()
        if not raw:
            return []

        # JSON list
        if raw.startswith("["):
            data = json.loads(raw)
            if not isinstance(data, list):
                raise ValueError(f"Expected JSON list in {p}")
            return [x for x in data if isinstance(x, dict)]

        # NDJSON
        out: list[dict[str, Any]] = []
        for i, line in enumerate(raw.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"Line {i} in {p} is not a JSON dict")
            out.append(obj)
        return out

    created: list[str] = []
    skipped_existing = 0
    downloaded = 0
    copied = 0
    skipped_no_url = 0

    for pat in patterns:
        for txt_path in sorted(directory.glob(pat)):
            alerts = _parse_alert_file(txt_path)

            for a in alerts:
                hp_url = a.get("healpix_url")
                if not hp_url:
                    skipped_no_url += 1
                    continue

                hp_url = str(hp_url)

                # URL -> download into directory (cached)
                if _is_url(hp_url):
                    fname = hp_url.split("/")[-1]
                    target = directory / fname

                    if target.exists() and not overwrite:
                        skipped_existing += 1
                        continue
                    local_path = _download_to(
                        hp_url,
                        directory,
                        timeout=timeout,
                        overwrite=overwrite,
                    )
                    created.append(local_path)
                    downloaded += 1
                    continue

                # local path -> optional copy into directory
                src = Path(hp_url)
                if src.exists() and src.is_file():
                    if copy_local_files:
                        dst = directory / src.name
                        if dst.exists() and not overwrite:
                            skipped_existing += 1
                        else:
                            dst.write_bytes(src.read_bytes())
                            created.append(str(dst))
                            copied += 1
                    continue

                # if local path is invalid, treat as "no url" effectively
                skipped_no_url += 1

    return {
        "directory": str(directory),
        "downloaded": downloaded,
        "copied": copied,
        "skipped_existing": skipped_existing,
        "skipped_no_url": skipped_no_url,
        "created": created,
    }

def _as_value(col: Any, unit: Optional[u.Unit] = None) -> np.ndarray:
    """
    Convert a column (with optional unit conversion) to a numpy array.
    Supports objects with a to_value() method, such as QTable, as well as
    plain arrays. Important for converting .txt files to skymaps.
    """
    if hasattr(col, "to_value"):
        return col.to_value(unit) if unit is not None else col.to_value()
    return np.asarray(col)

def _resolve_map_path(map_name: str, map_dir: str | Path) -> tuple[str, str]:
    """
    Returns (map_path, default_title)

    Supports:
    - URL -> download to map_dir
    - existing local file path (absolute or relative) -> use directly
    - filename relative to map_dir -> use map_dir/filename
    """
    if _is_url(map_name):
        map_path = _download_to(map_name, map_dir)
        fname = map_name.split("/")[-1]
        default_title = fname.split("_")[0]
        return map_path, default_title

    p = Path(map_name)
    if p.exists():
        # user passed a direct local path
        default_title = p.name.split("_")[0]
        return str(p), default_title

    # fallback: look in map_dir
    map_path = str(Path(map_dir) / map_name)
    default_title = Path(map_name).name.split("_")[0]
    return map_path, default_title

def load_icecube_skymaps(map_dir: str | Path = _P["map_dir"]) -> list[dict[str, Any]]:
    """
    Load local IceCube multiorder skymaps from `map_dir`.
    """
    p = Path(map_dir)
    if not p.exists():
        raise FileNotFoundError(str(p))

    out: list[dict[str, Any]] = []
    for fp in sorted(p.glob("*.fits.gz")):
        event_name = fp.name.split("_")[0] if "_" in fp.name else fp.stem
        out.append({"event_name": event_name, "healpix_url": str(fp)})

    return out

def _read_skymap_fix_coordsys(map_path: str, *, nest: bool = True) -> tuple[np.ndarray, Any]:
    """
    Fix a common issue with IceCube skymaps where the FITS COORDSYS keyword is written in lowercase.
    If the keyword is found in lowercase, it is rewritten in uppercase and the file is overwritten.
    Then, the skymap is re-read with the corrected header.
    """
    try:
        hpx, header = read_sky_map(map_path, nest=nest)
        return hpx, header
    except ValueError as e:
        data, header = fits.getdata(map_path, header=True)
        if "COORDSYS" in header and header["COORDSYS"] == "c":
            header["COORDSYS"] = "C"
            fits.writeto(map_path, data, header, overwrite=True)
            hpx, _ = read_sky_map(map_path, nest=nest)
            return hpx, header
        raise ValueError(e) from e
    
def _infer_alert_datetime_from_header(header: Any) -> Optional[str]:
    """
    Infer alert time from an IceCube skymap header 
    as GPS seconds in header key "gps_time" (IceCube public ROC skymaps)

    Returns UTC timestamp as ISO string (Time(...).isot) or None if not found.
    """
    if header is None:
        return None

    hdr = dict(header)

    # Scale "gps_time" to UTC
    if "gps_time" in hdr and hdr["gps_time"] is not None:
        try:
            return Time(float(hdr["gps_time"]), format="gps", scale="utc").isot
        except Exception:
            return None

    return None

# ------------------------------------------------------------------
# Plotting the skymap
# ------------------------------------------------------------------

SOURCE_COLORS = [
    "limegreen",   
    "magenta",     
    "gold",        
    "blueviolet",  
    "darkorange",  
    "royalblue",   
    "deeppink",    
]

def plot_IC_alert(
    map_name: str,
    *,
    map_dir: str = _P["map_dir"],
    coord: str = "icrs",
    view: Literal["full", "zoom"] = "full",
    r: float = 3.0,
    plot_source: Optional[dict[str, Any] | list[dict[str, Any]]] = None,
    p_region: Optional[float] = None,
    show_alert_datetime: bool = False,
    savepath: Optional[str] = None,
    show: bool = True,
) -> str:
        
    """
    Plot IceCube alert probability map with optional zoom view and alert datetime.

    Parameters:
    map_name (str): Map name or URL.
    map_dir (str, optional): Directory for local map storage.
    coord (str, optional): Coordinate system for plotting. 
    view (Literal["full", "zoom"], optional): View type. 
    r (float, optional): Zoom radius in degrees. 
    plot_source (Optional[dict[str, Any] | list[dict[str, Any]]], optional): LSST Candidate source(s) to plot. 
    p_region (Optional[float], optional): Cumulative probability region to highlight.
    show_alert_datetime (bool, optional): Show alert datetime in plot title.
    savepath (Optional[str], optional): Save plot to file.
    show (bool, optional): Show plot.

    Returns:
    str: Path to the plotted map.

    Notes:
    - Skymap header information is used to infer the alert time and center coordinates.
    """
    
    ligo_coord = {"E": "geo", "icrs": "astro", "G": "galactic"}
    astropy_frame = {"icrs": "icrs", "G": "galactic"}

    # --- load path + header
    map_path, default_title = _resolve_map_path(map_name, map_dir)
    title = default_title

    # read multi-order table (this is what you actually plot)
    skymap = QTable.read(map_path)

    # header for center + optional time
    # (fits header via read_sky_map is fine; we keep your COORDSYS fix)
    _hpx, header = _read_skymap_fix_coordsys(map_path, nest=True)  # internal detail, not user-facing
    header_dict = dict(header)

    # --- infer alert time (optional)
    alert_datetime = _infer_alert_datetime_from_header(header)

    # --- choose center from header RA/DEC (drop use_hpx_max)
    ra = float(header_dict["RA"]) * u.deg
    dec = float(header_dict["DEC"]) * u.deg
    center = SkyCoord(ra.to("rad"), dec.to("rad"), frame=astropy_frame.get(coord, "icrs"))

    # --- figure + axes layout
    fig = plt.figure(figsize=(6, 6), dpi=100)

    ax_inset = None
    if view == "full":
        ax = plt.axes(
            [0.05, 0.05, 0.8, 0.9],
            projection=f"{ligo_coord[coord]} degrees mollweide",
            center=SkyCoord(0 * u.rad, 0 * u.rad, frame=astropy_frame.get(coord, "icrs")),
        )
        ax_inset = plt.axes(
            [1.03, 0.3, 0.42, 0.42],
            projection=f"{ligo_coord[coord]} degrees zoom",
            center=center,
            radius=float(r) * u.deg,
        )

        if coord == "icrs":
            for key in ["ra", "dec"]:
                ax_inset.coords[key].set_ticks_visible(False)

        ax_inset.set_xlabel("")
        ax_inset.set_ylabel("")
        ax.grid()
        ax.locator_params(nbins=10)
        ax.mark_inset_axes(ax_inset)
        ax.connect_inset_axes(ax_inset, "upper left")
        ax.connect_inset_axes(ax_inset, "lower left")
        ax_inset.scalebar((0.1, 0.1), 1 * u.deg).label()
        ax_inset.compass(0.9, 0.1, 0.2)
        ax_inset.grid()

        ax.set_title(title)

        # inset title always RA/DEC (drop plot_zoom_title toggle)
        ax_inset.set_title(fr"RA: {ra.to_value(u.deg):.2f}$\degree$  DEC: {dec.to_value(u.deg):.2f}$\degree$")

    elif view == "zoom":
        ax = plt.axes(
            [0.05, 0.05, 0.8, 0.9],
            projection=f"{ligo_coord[coord]} degrees zoom",
            center=center,
            radius=float(r) * u.deg,
        )
        if coord == "icrs":
            for key in ["ra", "dec"]:
                ax.coords[key].set_ticks_visible(False)

        ax.set_xlabel("RA")
        ax.set_ylabel("DEC")
        ax.grid()

        # zoom view: RA/DEC as title (fixed behavior)
        ax.set_title(fr"RA: {ra.to_value(u.deg):.2f}$\degree$  DEC: {dec.to_value(u.deg):.2f}$\degree$")
    else:
        raise ValueError(f"Invalid view={view!r}, expected 'full' or 'zoom'.")

    ax_c = ax_inset if ax_inset is not None else ax

    # --- PROBDENSITY image
    prob_density = _as_value(skymap["PROBDENSITY"], u.deg**-2)
    m = mhp.HealpixMap(data=prob_density, uniq=skymap["UNIQ"], density=True)

    if view == "full":
        ax.imshow(m.get_wcs_img(ax), cmap="OrRd")
        im = ax_c.imshow(m.get_wcs_img(ax_c), cmap="OrRd")
        plt.colorbar(im, label=r"Prob density [$deg^{-2}$]")
    else:
        im = ax.imshow(m.get_wcs_img(ax), cmap="OrRd")
        plt.colorbar(im, label=r"Prob density [$deg^{-2}$]")

    # --- cumulative probability contours
    skymap.sort("PROBDENSITY", reverse=True)
    level, ipix = ah.uniq_to_level_ipix(skymap["UNIQ"])
    nside = ah.level_to_nside(level)
    pixel_area_sr = ah.nside_to_pixel_area(nside).to_value(u.sr)
    prob_density = _as_value(skymap["PROBDENSITY"])
    prob = pixel_area_sr * prob_density
    s = float(np.nansum(prob))
    if not np.isfinite(s) or s <= 0:
        raise ValueError(f"Invalid probability normalization for map: {map_path}")
    prob = prob / s
    cumprob = np.cumsum(prob)

    m_prob = mhp.HealpixMap(data=cumprob, uniq=skymap["UNIQ"], density=True)
    img_c = m_prob.get_wcs_img(ax_c)

    levels_styles = [(0.5, "black", "solid"), (0.9, "black", "dashed")]
    if p_region is not None:
        p = float(p_region)
        if 0.0 < p < 1.0 and p not in (0.5, 0.9):
            levels_styles.append((p, "cyan", "solid"))
    levels_styles.sort(key=lambda t: t[0])

    ax_c.contour(
        img_c,
        levels=[t[0] for t in levels_styles],
        colors=[t[1] for t in levels_styles],
        linestyles=[t[2] for t in levels_styles],
    )

    # --- area textbox only in full view
    if view == "full":
        i_50 = int(np.searchsorted(cumprob, 0.5))
        i_90 = int(np.searchsorted(cumprob, 0.9))

        area_50 = (pixel_area_sr[:i_50].sum() * u.sr).to(u.deg**2)
        area_90 = (pixel_area_sr[:i_90].sum() * u.sr).to(u.deg**2)

        lines = []
        if alert_datetime and show_alert_datetime:
            t = Time(alert_datetime, format="isot", scale="utc").to_value("datetime")
            lines.append(f"Alert UTC:\n {t:%Y-%m-%d}\n {t:%H:%M}\n\n")

        lines.append(f"Area 50%:\n {area_50:.3f} \n\nArea 90%:\n {area_90:.3f}")

        if p_region is not None:
            p = float(p_region)
            if 0.0 < p < 1.0 and p not in (0.5, 0.9):
                i_p = int(np.searchsorted(cumprob, p))
                area_p = (pixel_area_sr[:i_p].sum() * u.sr).to(u.deg**2)
                lines.append(f"\n\nArea {p*100:.0f}%:\n {area_p:.3f}")

        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        ax.text(1.9, 0.5, "".join(lines), fontsize=14, bbox=props, transform=ax.transAxes, va="center")
        
    # --- plot candidate source(s)
    if plot_source:
        sources = plot_source if isinstance(plot_source, list) else [plot_source]
        color_cycle = cycle(SOURCE_COLORS)
        handles = []
        for s in sources:
            color = next(color_cycle)
            name = str(s.get("name", s.get("id", "source")))
            ax_c.plot(float(s["ra"]), float(s["dec"]), "o", c=color, transform=ax_c.get_transform("world"))
            handles.append(Line2D([0], [0], marker="o", linestyle="None", label=name, color=color))
        if handles:
            ax_c.legend(handles=handles, framealpha=1, fontsize=8, loc="upper right")

    # --- save/show
    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return map_path


# ------------------------------------------------------------------
# Filters
# ------------------------------------------------------------------

def _extract_sources(sources: Any) -> list[dict[str, Any]]:
    """
    Normalize input to list of dicts:
      {"id": str, "ra": float, "dec": float, "name": str}

    Supported inputs:
    - (ra, dec) tuple/list
    - iterable of (ra, dec) tuples/lists
    - LSSTReport (pydantic) or dict-like with ["object"]["ra","dec","id"]
    - AmpelTransientReport (x.r.object.ra/dec/id)
    - AmpelReportSet (iterates over active reports)
    - iterable of any of the above
    """
    if sources is None:
        raise ValueError("sources must not be None")

    def from_radec(ra: float, dec: float) -> dict[str, Any]:
        ra_f = float(ra)
        dec_f = float(dec)
        sid = f"coord:{ra_f:.3f},{dec_f:.3f}"
        return {"id": sid, "ra": ra_f, "dec": dec_f, "name": sid}

    def one(x: Any) -> dict[str, Any]:
        # (ra, dec)
        if isinstance(x, (tuple, list)) and len(x) == 2 and all(isinstance(v, (int, float, np.number)) for v in x):
            return from_radec(x[0], x[1])

        # AmpelTransientReport-like
        if hasattr(x, "r") and hasattr(x.r, "object"):
            obj = x.r.object
            oid = str(getattr(obj, "id", "")).strip()
            ra = float(getattr(obj, "ra"))
            dec = float(getattr(obj, "dec"))
            sid = oid or f"coord:{ra:.3f},{dec:.3f}"
            return {"id": sid, "ra": ra, "dec": dec, "name": sid}

        # LSSTReport pydantic-like
        if hasattr(x, "object") and hasattr(x.object, "ra") and hasattr(x.object, "dec"):
            obj = x.object
            oid = str(getattr(obj, "id", "")).strip()
            ra = float(getattr(obj, "ra"))
            dec = float(getattr(obj, "dec"))
            sid = oid or f"coord:{ra:.3f},{dec:.3f}"
            return {"id": sid, "ra": ra, "dec": dec, "name": sid}

        # dict-like (either {"object": {...}} or directly {"ra":...,"dec":...})
        if isinstance(x, dict):
            obj = x.get("object", x)
            if isinstance(obj, dict) and ("ra" in obj) and ("dec" in obj):
                oid = str(obj.get("id", "")).strip()
                ra = float(obj["ra"])
                dec = float(obj["dec"])
                sid = oid or f"coord:{ra:.3f},{dec:.3f}"
                return {"id": sid, "ra": ra, "dec": dec, "name": sid}

        raise TypeError(f"Unsupported source element type: {type(x)!r}")

    # Treat strings/bytes as scalar (not iterable of sources)
    if isinstance(sources, (str, bytes)):
        raise TypeError("sources must not be str/bytes")

    # Single (ra,dec) or single report/dict -> wrap
    try:
        return [one(sources)]
    except TypeError:
        pass

    # Otherwise: iterable of elements
    if isinstance(sources, Iterable):
        out = [one(x) for x in sources]
        if not out:
            raise ValueError("sources iterable is empty")
        return out

    raise TypeError(f"Unsupported sources container type: {type(sources)!r}")



def select_pixels_sets(
    map_url: str,
    p_region: float = 0.9,
    map_dir: str = _P["map_dir"],
) -> dict[int, set[int]]:
    """
    Select HEALPix pixels covering top p_region probability mass.
    """
    map_path = _download_to(map_url, map_dir)

    skymap = QTable.read(map_path)
    skymap.sort("PROBDENSITY", reverse=True)

    level, ipix = ah.uniq_to_level_ipix(skymap["UNIQ"])
    nside = ah.level_to_nside(level)
    pixel_area_sr = ah.nside_to_pixel_area(nside).to_value(u.sr)
    prob_density = _as_value(skymap["PROBDENSITY"])
    prob = pixel_area_sr * prob_density
    s = float(np.nansum(prob))
    if not np.isfinite(s) or s <= 0:
        raise ValueError(f"Invalid probability normalization for map: {map_path}")
    prob = prob / s
    probsum = np.cumsum(prob)

    region_index = int(probsum.searchsorted(float(p_region)))

    out: dict[int, set[int]] = {}
    for ns_i, ip_i in zip(nside[:region_index], ipix[:region_index]):
        ns = int(ns_i)
        out.setdefault(ns, set()).add(int(ip_i))

    return out


def check_sources_in_map(
    sources: Any,
    map_url: str,
    p_region: float = 0.9,
    map_dir: str = _P["map_dir"],
    *,
    plot: bool = True,
    plot_kwargs: Optional[dict[str, Any]] = None,
) -> dict[str, bool]:
    """
    Check if a set of sources are inside a given probability region of a healpix map.

    Parameters:
    sources : List of sources to check, or a single source as dict.
    map_url : URL of the healpix map.
    p_region : Cumulative probability region to consider.
    map_dir : Directory for local map storage.
    plot : Plot the sources on the map.
    plot_kwargs : Keyword arguments for plot_IC_alert.

    Returns: Dictionary with source IDs as keys 
    and a boolean indicating if the source is inside the region as values.
    """
    src_list = _extract_sources(sources)
    pix_sets = select_pixels_sets(map_url, p_region=p_region, map_dir=map_dir)

    results: dict[str, bool] = {}
    for s in src_list:
        ra = float(s["ra"])
        dec = float(s["dec"])

        inside = False
        lon = ra * u.deg
        lat = dec * u.deg

        for ns, pixels in pix_sets.items():
            # compute pixel index for this nside in nested order
            ip = int(ah.lonlat_to_healpix(lon, lat, nside=int(ns), order="nested"))
            if ip in pixels:
                inside = True
                break

        results[str(s["id"])] = inside

    if plot:
        plot_IC_alert(
            map_url,
            map_dir=map_dir,
            view="full",
            p_region=p_region,
            plot_source=src_list,
            savepath=str(Path(map_dir) / "SourceChecks"),
            **plot_kwargs,
        )

    return results





if __name__ == "__main__":
    json_alerts = _P.get("json_alerts", "")
    file_dir = _P.get("file_dir", ".")

    if json_alerts:
        with open(os.path.join(file_dir, json_alerts), "r") as f:
            IC_alerts = json.load(f)

        for alert in IC_alerts:
            try:
                plot_IC_alert(alert["healpix_url"], show=False)
                plt.close()
            except IndexError as e:
                name = str(alert.get("event_name", "UNKNOWN"))
                print(f"Map of alert {name} has wrong shape. IndexError: {e}")


# ---------------------------------------------------------------------
# High-level wrapper
# ---------------------------------------------------------------------

class IceCubeAlert:
    """
    Convenience wrapper for a single IceCube alert.
    Encapsulates JSON parsing, healpix_url handling and basic plotting helpers.
    """
    def __init__(self, alert_dict: dict[str, Any]):
        """
        Initialize IceCubeAlert object.
        """
        self.raw = alert_dict

        self.healpix_url: str = str(alert_dict.get("healpix_url", ""))
        self.event_name: str = str(alert_dict.get("event_name", ""))
        self.alert_datetime: Optional[str] = alert_dict.get("alert_datetime")

        if not self.healpix_url:
            raise ValueError("IceCube alert has no 'healpix_url' field.")

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_kafka_string(cls, alert_str: str) -> "IceCubeAlert":
        """
        Create IceCubeAlert from raw Kafka JSON string.
        """
        try:
            alert_dict = json.loads(alert_str)
        except Exception as e:
            raise ValueError("Invalid JSON alert string.") from e

        return cls(alert_dict)

    @classmethod
    def from_dict(cls, alert_dict: dict[str, Any]) -> "IceCubeAlert":
        """
        Create IceCubeAlert from already parsed dict.
        """
        return cls(alert_dict)

    # ------------------------------------------------------------------
    # Plotting and Checking
    # ------------------------------------------------------------------

    def plot(
        self,
        *,
        zoom: bool = False,
        show: bool = True,
        show_alert_datetime: bool = False,
        savepath: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Plot this alert's HEALPix skymap.
        """
        return plot_IC_alert(
            self.healpix_url,
            view="zoom" if zoom else "full",
            show=show,
            show_alert_datetime=show_alert_datetime,
            savepath=savepath,
            **kwargs,
        )


    def contains_source(
        self,
        ra: float | None = None,
        dec: float | None = None,
        p_region: float = 0.9,
        *,
        sources: Any = None,
        plot: bool = True,
        **kwargs,
    ) -> bool | dict[str, bool]:

        """
        Check if a set of sources are inside a given probability region of this alert's HEALPix skymap.
        """
        if sources is None:
            if ra is None or dec is None:
                raise ValueError("Provide either sources=... or ra/dec")
            sources = (float(ra), float(dec))

        return check_sources_in_map(
            sources,
            self.healpix_url,
            p_region=p_region,
            plot=plot,
            plot_kwargs=kwargs,
        )

    def summary(self) -> dict[str, Any]:
        """
        Return minimal summary dict.
        """
        return {
            "event_name": self.event_name,
            "healpix_url": self.healpix_url,
            "alert_datetime": self.alert_datetime,
        }


