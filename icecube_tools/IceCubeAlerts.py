#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File:                icecube_tools.py
# License:             BSD-3-Clause
# Author:              cristozilleruelo
# Date:                11.2.2026
# Last Modified Date:  11.2.2026
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


# ---------------------------------------------------------------------
# Optional default params (kept for backwards compatibility)
# ---------------------------------------------------------------------


_P = {
    "map_dir": "./cache_ampelaccess/IceCubeAlerts",
    "save_dir": "./cache_ampelaccess/IceCubeAlerts",
    "file_dir": "./cache_ampelaccess/IceCubeAlerts",
    "json_alerts": "./cache_ampelaccess/IceCubeAlerts",
}


def _is_url(s: str) -> bool:
    return str(s).startswith("http://") or str(s).startswith("https://")


def _download_to(map_url: str, map_dir: str | Path) -> str:
    """
    Download map_url into map_dir and return local path.
    Overwrites existing file of same name.
    """
    file_name = map_url.split("/")[-1]
    map_dir = str(map_dir)
    os.makedirs(map_dir, exist_ok=True)
    map_path = os.path.join(map_dir, file_name)

    r = requests.get(map_url, timeout=30)
    r.raise_for_status()

    with open(map_path, "wb") as f:
        f.write(r.content)

    return map_path


class _HrefParser(HTMLParser):
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
    List absolute URLs of public IceCube ROC skymaps in a directory listing.

    Notes
    -----
    This returns only what is present in the given directory listing.
    If the directory contains only 2 maps, you'll only get 2 URLs.
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
) -> list[dict[str, str]]:
    """
    Convenience: return list of dicts compatible with your current alert schema usage:
      {"healpix_url": "<url>"}
    """
    return [{"healpix_url": u} for u in list_icecube_public_skymaps(index_url=index_url)]



def _read_skymap_fix_coordsys(map_path: str, *, nest: bool = True) -> tuple[np.ndarray, Any]:
    """
    Read skymap with a small robustness fix for COORDSYS='c' vs 'C'.
    Returns (hpx, header).
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

    Returns
    -------
    str | None
        UTC timestamp as ISO string (Time(...).isot) or None if not found.
    """
    if header is None:
        return None

    hdr = dict(header)

    # IceCube ROC convention: gps_time (seconds since GPS epoch)
    if "gps_time" in hdr and hdr["gps_time"] is not None:
        try:
            return Time(float(hdr["gps_time"]), format="gps", scale="utc").isot
        except Exception:
            return None

    return None



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
    map_dir: str = _P["map_dir"],
    coord: str = "icrs",
    nest: bool = True,
    plot_zoom_title: bool = True,
    use_hpx_max: bool = False,
    save_dir: str = _P["save_dir"],
    show_fig: bool = True,
    hpx: Optional[np.ndarray] = None,
    header: Optional[Any] = None,
    use_own_hpx: bool = False,
    normalize: bool = False,
    title: str = "",
    plot_source: Optional[dict[str, Any]] = None,
    r: float = 3,
    p_region: Optional[float] = None,
    debug: Optional[list[dict[str, Any]]] = None,
    view: Literal["full", "zoom"] = "full",
    alert_datetime: Optional[str] = None,
    show_alert_datetime: bool = False,
) -> Any:

    """
    Unified plotter for IceCube skymaps.

    view="full": Mollweide + zoom inset
    view="zoom": zoom-only view

    Returns
    -------
    map_path : str
      If debug is None
    (map_path, sel_pixels_lon, sel_pixels_lat) : tuple
      If debug is a list (quick inspection)
    """

    ligo_coord = {"E": "geo", "icrs": "astro", "G": "galactic"}
    astropy_frame = {"icrs": "icrs", "G": "galactic"}

    # --- load / read
    if not use_own_hpx:
        if _is_url(map_name):
            map_path = _download_to(map_name, map_dir)
            if not title:
                title = map_name.split("/")[-1].split("_")[0]
            hpx, header = _read_skymap_fix_coordsys(map_path, nest=nest)
        else:
            map_path = os.path.join(str(map_dir), str(map_name))
            if not title:
                title = str(map_name).split("_")[0]
            hpx, header = _read_skymap_fix_coordsys(map_path, nest=nest)
    else:
        if hpx is None or header is None:
            raise ValueError("use_own_hpx=True requires hpx and header to be provided.")
        if _is_url(map_name):
            map_path = _download_to(map_name, map_dir)
        else:
            map_path = os.path.join(str(map_dir), str(map_name))
        if not title:
            title = str(map_name).split("/")[-1].split("_")[0]

    if hpx is None or header is None:
        raise RuntimeError("Internal error: hpx/header not set.")
    
    # --- infer alert time from header if not provided
    if not alert_datetime:
        alert_datetime = _infer_alert_datetime_from_header(header)

    if alert_datetime is None:
        hdr = dict(header)
        cand = [k for k in hdr.keys() if any(t in k.upper() for t in ("DATE", "TIME", "MJD", "JD", "UTC", "TSTART", "TSTOP", "EVENT"))]
        print("DEBUG header time-like keys:", sorted(cand))
        for k in sorted(cand)[:40]:
            print(f"  {k} = {hdr.get(k)!r}")

    # --- normalize
    if normalize:
        hpx = np.asarray(hpx, dtype=float)
        hpx[hpx < 0] = 0.0
        s = float(np.sum(hpx))
        if s > 0:
            hpx = hpx / s

    # --- choose center
    header_dict = dict(header)
    h_ra = float(header_dict["RA"]) * u.deg
    h_dec = float(header_dict["DEC"]) * u.deg

    if use_hpx_max:
        arg_max = int(np.argmax(hpx))
        ra_deg, dec_deg = hp.pix2ang(int(header_dict["NSIDE"]), arg_max, lonlat=True, nest=True)
        ra = float(ra_deg) * u.deg
        dec = float(dec_deg) * u.deg
    else:
        ra = h_ra
        dec = h_dec

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
            radius=r * u.deg,
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

    elif view == "zoom":
        ax = plt.axes(
            [0.05, 0.05, 0.8, 0.9],
            projection=f"{ligo_coord[coord]} degrees zoom",
            center=center,
            radius=r * u.deg,
        )
        if coord == "icrs":
            for key in ["ra", "dec"]:
                ax.coords[key].set_ticks_visible(False)

        ax.set_xlabel("RA")
        ax.set_ylabel("DEC")
        ax.grid()
    else:
        raise ValueError(f"Invalid view={view!r}, expected 'full' or 'zoom'.")

    # Where to draw contours/sources: inset for full, main for zoom
    ax_c = ax_inset if ax_inset is not None else ax

    # --- multi-order table -> prob density image
    skymap = QTable.read(map_path)
    prob_density = skymap["PROBDENSITY"].to_value(u.deg**-2)
    m = mhp.HealpixMap(data=prob_density, uniq=skymap["UNIQ"], density=True)

    if view == "full":
        img_full = m.get_wcs_img(ax)
        ax.imshow(img_full, cmap="OrRd")
        img_inset = m.get_wcs_img(ax_c)
        im = ax_c.imshow(img_inset, cmap="OrRd")
        plt.colorbar(im, label=r"Prob density [$deg^{-2}$]")
    else:
        img = m.get_wcs_img(ax)
        im = ax.imshow(img, cmap="OrRd")
        plt.colorbar(im, label=r"Prob density [$deg^{-2}$]")

    # --- title
    if view == "full":
        ax.set_title(title)
    else:
        # zoom-only: keep old behavior (coord title vs map title)
        if plot_zoom_title:
            ax.set_title(fr"RA: {ra.to_value(u.deg):.2f}$\degree$  DEC: {dec.to_value(u.deg):.2f}$\degree$")
        else:
            ax.set_title(title)

    # --- cumulative prob contours
    skymap.sort("PROBDENSITY", reverse=True)
    level, ipix = ah.uniq_to_level_ipix(skymap["UNIQ"])
    nside = ah.level_to_nside(level)
    pixel_area = ah.nside_to_pixel_area(nside)
    prob = pixel_area * skymap["PROBDENSITY"]
    cumprob = np.cumsum(prob)

    i_50 = int(cumprob.searchsorted(0.5))
    i_90 = int(cumprob.searchsorted(0.9))

    m_prob = mhp.HealpixMap(data=cumprob, uniq=skymap["UNIQ"], density=True)
    img_c = m_prob.get_wcs_img(ax_c)

    levels_styles = [
        (0.5, "black", "solid"),
        (0.9, "black", "dashed"),
    ]
    if p_region is not None:
        p = float(p_region)
        if 0.0 < p < 1.0 and p not in (0.5, 0.9):
            levels_styles.append((p, "cyan", "solid"))

    levels_styles.sort(key=lambda t: t[0])
    levels = [t[0] for t in levels_styles]
    colors = [t[1] for t in levels_styles]
    linestyles = [t[2] for t in levels_styles]

    ax_c.contour(img_c, levels=levels, colors=colors, linestyles=linestyles)

    # --- area textbox only for full view (as before)
    if view == "full":
        area_50 = pixel_area[:i_50].sum().to(u.deg**2)
        area_90 = pixel_area[:i_90].sum().to(u.deg**2)

        lines = []
        if alert_datetime and show_alert_datetime:
            t = Time(alert_datetime, format="isot", scale="utc")
            dt = t.to_value("datetime")  # python datetime (UTC)
            date_str = dt.strftime("%Y-%m-%d")
            time_str = dt.strftime("%H:%M")

            lines.append(f"Alert UTC:\n {date_str}\n {time_str}\n\n")  # extra blank line after time

        lines.append(f"Area 50%:\n {area_50:.3f} \n\nArea 90%:\n {area_90:.3f}")


        if p_region is not None:
            p = float(p_region)
            if 0.0 < p < 1.0 and p not in (0.5, 0.9):
                i_p = int(cumprob.searchsorted(p))
                area_p = pixel_area[:i_p].sum().to(u.deg**2)
                lines.append(f"\n\nArea {p*100:.0f}%:\n {area_p:.3f}")

        text_str = "".join(lines)

        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        ax.text(1.9, 0.5, text_str, fontsize=14, bbox=props, transform=ax.transAxes,va="center")


    # --- plot candidate source(s)
    if plot_source:
        sources = plot_source if isinstance(plot_source, list) else [plot_source]
        color_cycle = cycle(SOURCE_COLORS)
        handles = []
        for s in sources:
            color = next(color_cycle)
            name = str(s.get("name", s.get("id", "source")))
            ax_c.plot(
                float(s["ra"]),
                float(s["dec"]),
                "o",
                c=color,
                transform=ax_c.get_transform("world"),
            )
            handles.append(Line2D([0], [0], marker="o", linestyle="None", label=name, color=color))

        if handles:
            ax_c.legend(handles=handles, framealpha=1, fontsize=8, loc="upper right")

    # --- zoom title on full inset (old behavior)
    if view == "full" and plot_zoom_title and ax_inset is not None:
        ax_inset.set_title(fr"RA: {ra.to_value(u.deg):.2f}$\degree$  DEC: {dec.to_value(u.deg):.2f}$\degree$")

    # --- debug pixels (only meaningful in zoom view; kept for convenience)
    sel_pixels_lat: list[float] = []
    sel_pixels_lon: list[float] = []
    if isinstance(debug, list):
        for dict_nside in debug:
            ns = int(dict_nside["nside"])
            pixels = dict_nside["pixels"]
            lon, lat = ah.healpix_to_lonlat(pixels, nside=ns, order="nested")
            sel_pixels_lat.extend(list(lat.to_value(u.deg)))
            sel_pixels_lon.extend(list(lon.to_value(u.deg)))

        ax_c.scatter(
            sel_pixels_lon,
            sel_pixels_lat,
            c="green",
            marker=".",
            s=3,
            transform=ax_c.get_transform("world"),
            alpha=0.5,
            label="Queried pixels",
        )
        ax_c.legend(fontsize=14)
        ax_c.set_title("Debug: Accepted pixels", fontsize=14)

    # --- saving
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        suffix = "_mw_zoom.png" if view == "zoom" else "_mw.png"
        plt.savefig(os.path.join(save_dir, f"{title}{suffix}"), bbox_inches="tight")

    if show_fig:
        plt.show()

    if isinstance(debug, list):
        return map_path, sel_pixels_lon, sel_pixels_lat
    return map_path


def set_new_alert(ic_alert: dict[str, Any], yaml_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Rewrite an Ampel job yaml template dict for a new IC alert.

    Parameters
    ----------
    ic_alert : dict
        JSON schema object from the GCN Kafka stream.
    yaml_dict : dict
        Parsed YAML template.

    Returns
    -------
    yaml_dict : dict
        Updated YAML dict.
    """
    map_name = ic_alert["healpix_url"].split("/")[-1]
    date_str = ic_alert["alert_datetime"].split("T")[0]

    params = yaml_dict["task"][0]["config"]["execute"][0]["config"]["execute"][0]["config"]

    params["map_name"] = map_name
    params["map_url"] = ic_alert["healpix_url"]
    params["date_str"] = date_str
    params["map_dir"] = "/Users/cristozilleruelo/Ampel-HU-astro/RubiCube/ic_alerts_temp"

    yaml_dict["task"][0]["config"]["execute"][0]["config"]["execute"][0]["config"] = params

    stream_list = yaml_dict["task"][1]["config"]["supplier"]["config"]["loader"]["config"]["stream"].split("_")
    stream_list.pop(0)
    new_name = "%%" + map_name.split("_")[0]
    stream_list.insert(0, new_name)
    yaml_dict["task"][1]["config"]["supplier"]["config"]["loader"]["config"]["stream"] = "_".join(stream_list)

    if isinstance(ic_alert.get("event_name"), list):
        ic_alert["event_name"] = ic_alert["event_name"][0]

    file_name = str(ic_alert["event_name"]) + ".pdf"
    pdf_path = yaml_dict["task"][3]["config"]["stage"]["config"]["execute"][0]["config"]["pdf_path"]
    directory = os.path.join(*pdf_path.split("/")[:-1])
    yaml_dict["task"][3]["config"]["stage"]["config"]["execute"][0]["config"]["pdf_path"] = os.path.join(os.sep, directory, file_name)

    return yaml_dict


def write_new_yaml(
    ic_alert: dict[str, Any] | str,
    yaml_name: str,
    save_name: str = "",
    save_dir: str = _P["file_dir"],
) -> str:
    """
    Write an updated Ampel job YAML for a new IC alert.

    Parameters
    ----------
    ic_alert : dict | str
        IC alert schema dict or JSON string.
    yaml_name : str
        Template YAML filename in save_dir.
    save_name : str
        Output filename; if empty, overwrites yaml_name.
    save_dir : str
        Directory containing templates / where output is written.

    Returns
    -------
    path : str
        Path to written YAML file.
    """
    import yaml  # local import to keep module import light

    with open(os.path.join(save_dir, yaml_name), "r") as f:
        yaml_dict = yaml.safe_load(f)

    if isinstance(ic_alert, str):
        ic_alert = json.loads(ic_alert)

    new_yaml = set_new_alert(ic_alert, yaml_dict)

    if save_name == "":
        save_name = yaml_name

    out_path = os.path.join(save_dir, save_name)
    with open(out_path, "w") as f:
        yaml.dump(new_yaml, f, sort_keys=False)

    return out_path

def _extract_sources(sources: Any) -> list[dict[str, Any]]:
    """
    Normalize input to list of dicts:
      {"id": str, "ra": float, "dec": float, "name": str}

    Supported inputs
    ----------------
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
    nest: bool = True,
) -> dict[int, set[int]]:
    """
    Select HEALPix pixels covering top p_region probability mass.

    Returns
    -------
    dict: nside -> set(pixel_index)
    """
    map_path = _download_to(map_url, map_dir)
    _hpx, _header = _read_skymap_fix_coordsys(map_path, nest=nest)

    skymap = QTable.read(map_path)
    skymap.sort("PROBDENSITY", reverse=True)

    level, ipix = ah.uniq_to_level_ipix(skymap["UNIQ"])
    nside = ah.level_to_nside(level)
    pixel_area = ah.nside_to_pixel_area(nside)
    prob = pixel_area * skymap["PROBDENSITY"]
    cumprob = np.cumsum(prob)

    region_index = int(cumprob.searchsorted(float(p_region)))

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
    nest: bool = True,
    *,
    plot: bool = True,
    plot_kwargs: Optional[dict[str, Any]] = None,
) -> dict[str, bool]:
    """
    Check multiple sources against the p_region containment region.

    Parameters
    ----------
    sources:
      see _extract_sources()
    plot:
      If True, plot once and mark all sources.

    Returns
    -------
    dict: object_id (string) -> in_region (bool)
    """
    src_list = _extract_sources(sources)
    pix_sets = select_pixels_sets(map_url, p_region=p_region, map_dir=map_dir, nest=nest)

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
        # mark all sources in ONE plot
        plot_kwargs = plot_kwargs or {}
        plot_IC_alert(
            map_url,
            map_dir=map_dir,
            save_dir=str(Path(map_dir) / "SourceChecks"),
            p_region=p_region,
            plot_source=src_list,   # see next section (plot_IC_alert patch)
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
                plot_IC_alert(alert["healpix_url"], show_fig=False, print_radec_diff=True)
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

    Encapsulates:
    - JSON parsing (from Kafka string or dict)
    - healpix_url handling
    - basic plotting helpers

    Designed for minimal notebook usage.
    """

    def __init__(self, alert_dict: dict[str, Any]):
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
    # Core functionality
    # ------------------------------------------------------------------

    def plot(
        self,
        *,
        zoom: bool = False,
        show: bool = True,
        show_alert_datetime: bool = False,
        **kwargs,
    ) -> str:
        """
        Plot this alert's HEALPix skymap.

        zoom=False -> full (mollweide + inset)
        zoom=True  -> zoom-only
        """
        return plot_IC_alert(
            self.healpix_url,
            view="zoom" if zoom else "full",
            show_fig=show,
            alert_datetime=self.alert_datetime,
            show_alert_datetime=show_alert_datetime,
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


    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """
        Return minimal summary dict.
        """
        return {
            "event_name": self.event_name,
            "healpix_url": self.healpix_url,
            "alert_datetime": self.alert_datetime,
        }

    def __repr__(self) -> str:
        return (
            f"IceCubeAlert(event_name={self.event_name!r}, "
            f"datetime={self.alert_datetime!r})"
        )

