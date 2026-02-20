#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File:                AmpelReport.py
# License:             BSD-3-Clause
# Author:              jno
# Date:                19.1.2026
# Last Modified Date:  19.2.2026
# Last Modified By:    Felix Fischer

"""
AmpelReport
===========

Utilities for visualizing and summarizing individual AMPEL alert reports.

This module provides:
- Lightcurve plotting
- Classification probability visualization
- Host/environment information display
- Finder stamp retrieval from CDS surveys

The main entry point is the `AmpelTransientReport` class, which wraps
an `LSSTReport` object from ampel-hu-astro.

"""


from __future__ import annotations
from typing import Optional, Any
import io
import requests

import numpy as np
import pandas as pd

from astropy.time import Time

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.pyplot import Figure
from matplotlib.gridspec import GridSpec

from .LSSTReportModel import LSSTReport

# ------------------------------------------------------------
# Finder stamp (Thumbnails)
# ------------------------------------------------------------

def get_finder_stamp(
    ra: float,
    dec: float,
    size: int = 240,
    surveys: list[str] | None = None,
    timeout: float = 8.0,
    fov_arcsec: float = 45.0,
) -> tuple[np.ndarray | None, str | None]:
    """
    Best-effort finder stamp: CDS hips2fits JPEG

    Returns (image_array, label). image_array is 2D float array or None.
    """

    if surveys is None:
        surveys = [
            "CDS/P/DESI-Legacy-Surveys/DR10/color",
            "CDS/P/DECaLS/DR5/color",
            "CDS/P/DES-DR2/ColorIRG",
            "CDS/P/Skymapper/DR4/color",
            "CDS/P/DSS2/color",
        ]

    labels = {
        "CDS/P/DESI-Legacy-Surveys/DR10/color": "DESI DR10",
        "CDS/P/DES-DR2/ColorIRG": "DES",
        "CDS/P/Skymapper/DR4/color": "SkyMapper",
        "CDS/P/DECaLS/DR5/color": "DECaLS",
        "CDS/P/DSS2/color": "DSS2",
    }

    BASE_URLS = [
    "https://alasky.cds.unistra.fr/hips-image-services/hips2fits",
    "https://alaskybis.cds.unistra.fr/hips-image-services/hips2fits",
    ]

    fov_deg = float(fov_arcsec) / 3600.0

    try:
        from PIL import Image, ImageOps  # noqa: PLC0415
    except Exception:
        return None, None

    for hips in surveys:
        try:
            params: dict[str, str | int | float] = {
                "hips": hips,
                "ra": ra,
                "dec": dec,
                "fov": fov_deg,
                "width": size,
                "height": size,
                "format": "jpg",
                "projection": "TAN",
            }
            for base_url in BASE_URLS:
                r = requests.get(base_url, params=params, timeout=timeout)

            if r.status_code != 200 or not r.content:
                continue

            img = Image.open(io.BytesIO(r.content))
            img = ImageOps.exif_transpose(img).convert("RGB")
            arr = np.asarray(img).astype(float)

            # RGB -> grayscale [0,1]
            arr = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
            arr /= 255.0

            return arr, labels.get(hips, hips)

        except Exception as e:
            print(f"[finder] hips={hips} failed: {type(e).__name__}: {e}")
            continue

    return None, None


def normalize_image_for_imshow(arr: np.ndarray) -> np.ndarray:
    """
    Normalize an image array to the range [0,1] for display with imshow.

    The image array is assumed to be in one of the following formats:
    - 2D grayscale image
    - 3D RGB image, either as (height, width, 3) or (3, height, width)
    - 3D RGBA image, either as (height, width, 4) or (4, height, width)

    If the image array has an unsupported shape, a TypeError is raised.
    """
    a = np.asarray(arr)

    if a.ndim == 2:
        return a

    if a.ndim == 3 and a.shape[2] in (3, 4):
        rgb = a[..., :3].astype(float)
        return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]

    if a.ndim == 3 and a.shape[0] in (3, 4):
        rgb = a[:3, ...].astype(float)
        return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]

    if a.ndim == 3:
        return a[0, ...] if a.shape[0] < a.shape[-1] else a[..., 0]

    raise TypeError(f"Unsupported stamp shape for imshow: {a.shape}")

def add_gap_crosshair(
    ax, *, gap_frac: float = 0.2, lw: float = 1.0, alpha: float = 0.8
) -> None:
    """
    Draw a crosshair on the finder that leaves a central gap.
    gap_frac=0.2 leaves the central 20% of the axis free in both directions.
    """
    if not (0.0 < gap_frac < 1.0):
        return

    g0 = 0.5 - gap_frac / 2.0
    g1 = 0.5 + gap_frac / 2.0

    # Vertical segments (x=0.5), leaving a gap [g0, g1]
    ax.plot(
        [0.5, 0.5], [0.0, g0], transform=ax.transAxes, lw=lw, alpha=alpha, color="white"
    )
    ax.plot(
        [0.5, 0.5], [g1, 1.0], transform=ax.transAxes, lw=lw, alpha=alpha, color="white"
    )

    # Horizontal segments (y=0.5)
    ax.plot(
        [0.0, g0], [0.5, 0.5], transform=ax.transAxes, lw=lw, alpha=alpha, color="white"
    )
    ax.plot(
        [g1, 1.0], [0.5, 0.5], transform=ax.transAxes, lw=lw, alpha=alpha, color="white"
    )

# ------------------------------------------------------------
# Spider diagram for classification
# ------------------------------------------------------------

def clean_classprob_label(label: str) -> str:
    """
    Strip "P(...)" wrapper from class labels, for cleaner display.
    """
    if label.startswith("P(") and label.endswith(")"):
        return label[2:-1]
    return label

 
def create_classprob_radar(
        classprobs: dict[str, float],
        ax: Axes,
        threshold: float = 0.01,
        title: str = "Class probabilities",
    ):
        """
        Radar chart for class probabilities.
        Shows all classes with p >= threshold.
        Gracefully handles N<3 via normal bar plot fallback (non-polar).
        """
        # Filter + sort (descending)
        items = [(k, float(v)) for k, v in classprobs.items() if float(v) >= threshold]
        items.sort(key=lambda kv: kv[1], reverse=True)

        if not items:
            ax.axis("off")
            ax.text(
                0.5, 0.5,
                "No classes >= {:.2f}".format(threshold),
                ha="center", va="center", fontsize="small",
            )
            return ax

        labels = [clean_classprob_label(k) for k, _ in items]
        vals = np.array([v for _, v in items], dtype=float)

        n = len(vals)

        # Fallback: for less than 3 produce a normal bar plot
        if n < 3:
            fig = ax.figure
            pos = ax.get_position()

            # Remove polar axis
            ax.remove()

            axb = fig.add_axes(pos)  # normal cartesian axes
            axb.bar(range(n), vals, alpha=0.6, edgecolor="black", linewidth=0.8)

            axb.set_title(title, fontsize=11, pad=6)
            axb.set_ylim(0.0, 1.0)
            axb.set_xticks(range(n))
            axb.set_xticklabels(labels, fontsize=11)
            axb.set_ylabel("p", fontsize=10)
            axb.grid(True, axis="y", alpha=0.3)

            return axb

        # Standard radar polygon (polar)
        ax.set_title(title, fontsize=11, pad=8)
        ax.set_ylim(0.0, 1.08)
        ax.spines["polar"].set_visible(False)
        rmax = 1
        theta_full = np.linspace(0, 2*np.pi, 361)
        ax.plot(theta_full, np.full_like(theta_full, rmax), color="black", lw=1.0, zorder=10)
        rticks = [0.5, 1.0]
        ax.set_yticks(rticks)
        ax.set_yticklabels([str(x) for x in rticks], fontsize=9, alpha=0.7)
        ax.tick_params(axis="y", pad=4)

        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        theta_closed = np.r_[theta, theta[0]]
        vals_closed = np.r_[vals, vals[0]]

        ax.set_theta_offset(np.pi / 2.0)
        ax.set_theta_direction(-1)

        ax.plot(theta_closed, vals_closed, linewidth=1.2)
        ax.fill(theta_closed, vals_closed, alpha=0.25)

        ax.set_xticks(theta)
        ax.set_xticklabels(labels, fontsize=11)
        ax.grid(True, alpha=0.4)

        return ax

# ------------------------------------------------------------
# Small Helper
# ------------------------------------------------------------

def _jd_to_ymdhm(jd: float) -> str:
    """
    Convert Julian Date (JD) to human-readable UTC string.
    JD -> 'YYYY-MM-DD HH:MM' (UTC)
    """
    try:
        t = Time(float(jd), format="jd", scale="utc")
        return t.iso[:16]
    except Exception:
        return "?"


# Keys as provided by tabulators tied to labels and colors
BANDINFO = {
        # ZTF
        "ztfg": {"label": "ZTF g", "c": "green"},
        "ztfr": {"label": "ZTF r", "c": "red"},
        "ztfi": {"label": "ZTF i", "c": "orange"},

        # LSST
        "lsstu": {"label": "LSST u", "c": "purple"},
        "lsstg": {"label": "LSST g", "c": "blue"},
        "lsstr": {"label": "LSST r", "c": "green"},
        "lssti": {"label": "LSST i", "c": "orange"},
        "lsstz": {"label": "LSST z", "c": "red"},
        "lssty": {"label": "LSST y", "c": "darkred"},
}


class AmpelTransientReport():
    """
    Convenience wrapper around an `LSSTReport` for visualization and inspection.

    As a parameter, it takes an `LSSTReport` object, which is the standard format for AMPEL alert reports.
    It contains object metadata, photometry, classifications, and host information.
    """

    r: LSSTReport
    phot_table: None
    catalog_thumbnail = {'stamp':None, 'label':None}

    def __init__(self, report: LSSTReport):
        """
        Initialize an AmpelTransientReport from an LSSTReport.
        If the input is a dictionary, it is validated against the LSSTReport model.
        """
        if isinstance(report, LSSTReport):
            self.r = report
        else:
            self.r = LSSTReport.model_validate(report)
        
        self.phot_table = None 
        
    # ------------------------------------------------------------
    # Inspect the Report
    # ------------------------------------------------------------

    def create_table(self, backup_zp=31.4):
        """
        Convert existing list of Photometric Points to pandas table.

        Assumes photometry datapoints contain:
        - time (JD)
        - flux
        - fluxerr
        - zp
        - band
        """

        if len(self.r.photometry) == 0:
            self.phot_table = None
            return

        # pydantic objects -> list[dict] for DataFrame
        rows = [p.model_dump() for p in self.r.photometry]
        self.phot_table = pd.DataFrame.from_records(rows)


        # If phot points use different zeropoints, shift to backup value.
        if len(set(self.phot_table["zp"]))>1:
            scaling = 10**( 2.5* ( backup_zp - self.phot_table["zp"] ) )
            self.phot_table['flux'] *= scaling
            self.phot_table['fluxerr'] *= scaling
            self.phot_table['zp'] = backup_zp

    
    def to_summary_dict(
        self,
        snr_limit: float = 5.0,
        cls_threshold: float = 0.01,
    ) -> dict[str, Any]:
        """
        Helper method for `inspect()`: extract a human-friendly summary dict.
        Get key information about the object, photometry, classifications, and host.
        """


        r = self.r

        # Object basics
        obj = r.object
        obj_id = obj.id
        ext_id = obj.external_id
        ra = obj.ra
        dec = obj.dec


        # Photometry summary
        phot = r.photometry
        phot_summary: dict[str, Any] = {
            "n_points": 0,
            "bands": [],
            "time_jd_min": None,
            "time_jd_max": None,
            "first_utc": None,
            "last_utc": None,
            "n_det": 0,
            "n_det_by_band": {},
        }

        if len(phot) > 0:
            # build a small DF for robust stats
            df = pd.DataFrame.from_records([p.model_dump() for p in phot]).copy()
            for c in ("time", "flux", "fluxerr"):
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            if "band" in df.columns:
                df["band"] = df["band"].astype(str)

            df = df.dropna(subset=["time"])
            phot_summary["n_points"] = int(len(df))

            if len(df) > 0:
                tmin = float(df["time"].min())
                tmax = float(df["time"].max())
                phot_summary["time_jd_min"] = tmin
                phot_summary["time_jd_max"] = tmax
                phot_summary["first_utc"] = _jd_to_ymdhm(tmin)
                phot_summary["last_utc"] = _jd_to_ymdhm(tmax)

                if "band" in df.columns:
                    phot_summary["bands"] = sorted(df["band"].dropna().unique().tolist())

                # detections by SNR
                if "flux" in df.columns and "fluxerr" in df.columns:
                    ok = np.isfinite(df["flux"]) & np.isfinite(df["fluxerr"]) & (df["fluxerr"] > 0)
                    df2 = df.loc[ok].copy()
                    if len(df2) > 0:
                        snr = np.abs(df2["flux"].to_numpy(dtype=float)) / df2["fluxerr"].to_numpy(dtype=float)
                        det = snr >= float(snr_limit)
                        phot_summary["n_det"] = int(det.sum())

                        if "band" in df2.columns:
                            det_by_band: dict[str, int] = {}
                            for b in sorted(df2["band"].dropna().unique()):
                                m = (df2["band"] == b).to_numpy()
                                det_by_band[str(b)] = int(np.sum(det[m]))
                            phot_summary["n_det_by_band"] = det_by_band


        # Classification summary
        cls_threshold_f = float(cls_threshold)
        classifications: list[dict[str, Any]] = []

        for c in (r.classification or []):
            c_entry: dict[str, Any] = {
                "name": getattr(c, "name", None),
                "version": getattr(c, "version", None),
                "info": getattr(c, "info", None),
                "models": [],
            }

            for m in (getattr(c, "models", None) or []):
                probs = (getattr(m, "probabilities", None) or {})
                probs_f = {str(k): float(v) for k, v in probs.items()}

                items = [
                    (clean_classprob_label(str(k)), float(v))
                    for k, v in probs_f.items()
                    if np.isfinite(v) and float(v) >= cls_threshold_f
                ]
                items.sort(key=lambda kv: kv[1], reverse=True)

                c_entry["models"].append({
                    "model": getattr(m, "model", None),
                    "probs": probs_f,   # raw dict
                    "top": items,       # thresholded + sorted convenience
                })

            classifications.append(c_entry)


        # Host summary
        host_list = r.host
        host_summary = {"has_host": False, "redshift": None, "distance_arcsec": None, "info": None}

        if len(host_list) > 0:
            h0 = host_list[0]
            host_summary["has_host"] = True
            host_summary["redshift"] = float(h0.redshift) if h0.redshift is not None else None
            host_summary["distance_arcsec"] = float(h0.distance) if h0.distance is not None else None
            host_summary["info"] = h0.info


        return {
            "object": {"id": obj_id, "external_id": ext_id, "ra": ra, "dec": dec},
            "photometry": phot_summary,
            "classifications": classifications,
            "host": host_summary,
        }



    def inspect(
        self,
        snr_limit: float = 5.0,
        cls_threshold: float = 0.01,
    ) -> dict[str, Any]:
        """
        Human-friendly one-shot overview.
        Prints a compact summary and returns a dict with the same content.
        """
        r = self.r
        out = self.to_summary_dict(
            snr_limit=snr_limit,
            cls_threshold=cls_threshold,
        )

        # Optional: top-level keys (quick sanity)
        keys = sorted(list(r.model_fields.keys()))
        print(f"Top-level keys ({len(keys)}): {', '.join(keys)}")


        obj = out["object"]
        print(f"Object: id={obj.get('id')}  external_id={obj.get('external_id')}")
        if obj.get("ra") is not None and obj.get("dec") is not None:
            print(f"  RA/Dec: {float(obj['ra']):.5f}, {float(obj['dec']):.5f} deg")

        ph = out["photometry"]
        print(f"Photometry: {ph['n_points']} pts  bands={ph['bands']}")
        if ph["first_utc"] and ph["last_utc"]:
            print(f"  time: {ph['first_utc']} .. {ph['last_utc']} UTC")
        if ph["n_points"] > 0:
            print(f"  detections (SNR≥{snr_limit:g}): {ph['n_det']}  by band: {ph['n_det_by_band']}")

        cls_all = out.get("classifications", [])
        if cls_all:
            for i, c in enumerate(cls_all):
                name = c.get("name", "?")
                ver = c.get("version", "?")
                info = c.get("info", None)
                print(f"Classification [{i}]: {name} (v{ver})" + (f"  info={info!r}" if info else ""))

                models = c.get("models", []) or []
                if not models:
                    print("  (no models)")
                    continue

                for j, m in enumerate(models):
                    mname = m.get("model", "?")
                    print(f"  Model ({j}): {mname}")

                    top = m.get("top", []) or []
                    if top:
                        for k, v in top:
                            print(f"    {k:12s}  p={v:.3f}")
                    else:
                        print(f"    (no classes ≥ {cls_threshold:g})")
        else:
            print("Classification: none")


        host = out["host"]
        if host["has_host"]:
            z = host["redshift"]
            d = host["distance_arcsec"]
            print("Host:")
            if z is not None:
                try:
                    print(f"  z = {float(z):.4f}")
                except Exception:
                    print(f"  z = {z!r}")
            if d is not None:
                try:
                    print(f"  distance = {float(d):.2f} arcsec")
                except Exception:
                    print(f"  distance = {d!r}")
            if host.get("info") is not None:
                print(f"  info: {host.get('info')!r}")
        else:
            print("Host: none")

        return out

    # ------------------------------------------------------------
    # Finder stamp (Thumbnails)
    # ------------------------------------------------------------
    
    def get_catalogimage(self, surveys: Optional[list[str]] = None):
        """
        Load a catalog thumbnail from CDS by using the helper functions.
        """

        stamp, stamp_label = get_finder_stamp(
            ra=self.r.object.ra, dec=self.r.object.dec, 
            size=240, fov_arcsec=60.0, surveys = surveys
        )
        if stamp is None:
            # Set to -1 to indicate we tried but failed
            self.catalog_thumbnail = {'stamp':-1, 'label':'fail'}
        else:
            self.catalog_thumbnail = {
                'stamp':normalize_image_for_imshow(stamp), 
                'label':stamp_label
            }

    # ------------------------------------------------------------
    # Classification and Classifier
    # ------------------------------------------------------------

    def list_classifiers(self) -> list[dict[str, Any]]:
        """
        Return a notebook-friendly summary of available classifications/models.
        """
        out: list[dict[str, Any]] = []
        for ci, c in enumerate(self.r.classification):
            cinfo = {
                "ci": ci,
                "name": getattr(c, "name", None),
                "version": getattr(c, "version", None),
                "info": getattr(c, "info", None),
                "models": [],
            }
            for mi, m in enumerate(getattr(c, "models", []) or []):
                probs = getattr(m, "probabilities", {}) or {}
                keys = sorted([clean_classprob_label(str(k)) for k in probs.keys()])
                cinfo["models"].append({
                    "mi": mi,
                    "model": getattr(m, "model", None),
                    "n_classes": len(keys),
                    "classes": keys,
                })
            out.append(cinfo)
        return out
    
    def print_classifiers(self, max_classes: int = 30) -> None:
        """
        Printer for the list of classifiers/models, with a compact summary of available classes.
        """
        items = self.list_classifiers()
        print(f"n_classifications = {len(items)}")
        for c in items:
            print(f"\n[{c['ci']}] {c['name']}  v={c['version']}  info={c['info']!r}")
            for m in c["models"]:
                cls = m["classes"]
                tail = "" if len(cls) <= max_classes else f" (+{len(cls)-max_classes} more)"
                print(f"  ({m['mi']}) model={m['model']!r}  n_classes={m['n_classes']}")
                print("      " + ", ".join(cls[:max_classes]) + tail)

    def get_class_probabilities(self, classifier=0, model=0):
        """
        Select a classifier (by index or by name) and model index and return probabilities dict.
        If there are no probabilities, return an empty dict.
        """
        cls_list = getattr(self.r, "classification", None)

        # Robust fallback
        if not cls_list:
            return {}

        if isinstance(classifier, int):
            ci = classifier
            if ci < 0 or ci >= len(cls_list):
                return {}
            c = cls_list[ci]
        else:
            matches = [c for c in cls_list if getattr(c, "name", None) == classifier]
            if not matches:
                return {}
            c = matches[0]

        models = getattr(c, "models", None) or []
        if model < 0 or model >= len(models):
            return {}

        m = models[model]
        probs = getattr(m, "probabilities", None) or {}
        # ensure plain dict[str,float]-ish
        out = {}
        for k, v in probs.items():
            try:
                out[str(k)] = float(v)
            except Exception:
                continue
        return out


    def show_classification(self, ax=None, classifier=0, model=0, threshold=0.0, **kwargs):
        """
        Show a radar plot of class probabilities for a given classifier/model.
        If there is no classification at all, show a placeholder.
        """
        cls_list = getattr(self.r, "classification", None)

        # create ax if needed (polar)
        if ax is None:
            fig = plt.figure(figsize=(4, 4), dpi=100)
            ax = fig.add_subplot(111, projection="polar")

        # helper: show placeholder
        def _placeholder(msg: str) -> Any:
            ax.set_axis_off()
            ax.text(
                0.5, 0.5, msg,
                transform=ax.transAxes,
                ha="center", va="center",
                fontsize=10,
            )
            return ax

        # robust fallback: no classification at all
        if not cls_list:
            return _placeholder("No classification")

        # pick classifier
        if isinstance(classifier, int):
            ci = classifier
            if ci < 0 or ci >= len(cls_list):
                return _placeholder(f"No classifier idx {ci}")
            c = cls_list[ci]
        else:
            matches = [c for c in cls_list if getattr(c, "name", None) == classifier]
            if not matches:
                return _placeholder(f"No classifier '{classifier}'")
            c = matches[0]

        models = getattr(c, "models", None) or []
        if model < 0 or model >= len(models):
            return _placeholder(f"No model idx {model}")

        m = models[model]
        title = f"{getattr(c,'name','')} (v{getattr(c,'version','')}): {getattr(m,'model','')}"

        probs_raw = getattr(m, "probabilities", None) or {}
        probs = {}
        for k, v in probs_raw.items():
            try:
                probs[str(k)] = float(v)
            except Exception:
                continue

        # optional: if probs empty, also placeholder (depends on your taste)
        if not probs:
            return _placeholder("No probabilities")

        return create_classprob_radar(probs, ax, threshold=threshold, title=title)


    # ------------------------------------------------------------
    # Info Block
    # ------------------------------------------------------------
    
    def show_hostinfo(
        self, 
        fig: Optional[Figure] = None, 
        classprobs: Optional[dict[str, float]] = None,
        cls_threshold: float = 0.01,
        cls_max_items: int = 12,
        classifier: int | str = 0,
        model: int = 0
    ) -> Figure:
        """
        Produce a figure showing information about class probabilities, the object and host, alongside a finder stamp if available.
        """
        
        if fig is None:
            fig = plt.figure(figsize=(8, 5))
            gs = GridSpec(1, 2, figure=fig, width_ratios=[1.2, 1])
            ax_text = fig.add_subplot(gs[0])
            ax_img = fig.add_subplot(gs[1])
        else:
            gs = GridSpec(1, 2, figure=fig, width_ratios=[1.2, 1])
            ax_text = fig.add_subplot(gs[0])
            ax_img = fig.add_subplot(gs[1])

        if self.catalog_thumbnail["stamp"] is None:
            self.get_catalogimage()

        ax_text.axis("off")

        lines: list[str] = []

        # ---- Class probs ----
        if classprobs is None:
            classprobs = self.get_class_probabilities(classifier=classifier, model=model) 

        if classprobs:
            items = [(clean_classprob_label(k), float(v)) for k, v in classprobs.items()]
            items = [(k, v) for k, v in items if v >= cls_threshold]
            items.sort(key=lambda kv: kv[1], reverse=True)

            if items:
                lines.append(f"Class Probabilities (p>{cls_threshold:.2f}):")
                for k, v in items[:cls_max_items]:
                    lines.append(f"{k}: {v:.3f}")
                lines.append("------------------------")

        # ---- Host info ----
        host = self.r.host



        if len(host) > 0:
            lines.append("Host attributes:")
            h0 = host[0]

            z = h0.redshift
            if z is not None:
                lines.append(f"Redshift: {float(z):.4f}")

            dist = h0.distance
            if dist is not None:
                lines.append(f"Host distance: {float(dist):.2f} arcsec")

            info = h0.info
            if info is not None:
                lines.append(f"Info: {info}")
        
            lines.append("---------------------------")
        
        # ---- Object info ----
        lines.append("Object attributes:")

        phot = self.r.photometry
        first_str, last_str = None, None
        if isinstance(phot, list) and len(phot) > 0:
            try:
                times = [float(p.time) for p in phot if p.time is not None]
                times = [float(t) for t in times]
                if len(times) > 0:
                    first_str = _jd_to_ymdhm(min(times))
                    last_str  = _jd_to_ymdhm(max(times))
            except Exception:
                pass
        if first_str is not None and last_str is not None:
            lines.append(f"First phot: {first_str} UTC")
            lines.append(f"Last phot:  {last_str} UTC")
    
        ra = self.r.object.ra
        dec = self.r.object.dec
        if ra is not None and dec is not None:
            lines.append(f"RA: {float(ra):.4f} deg")
            lines.append(f"Dec: {float(dec):.4f} deg")

        if lines:
            ax_text.text(
                0.0, 1.0,
                "\n".join(lines),
                va="top", ha="left",
                fontsize=10,
                transform=ax_text.transAxes,
            )
            ax_text.set_title("Info", fontsize=11, pad=10, loc="left")
        
        # ---- Finder ----
        stamp = self.catalog_thumbnail.get("stamp")

        if isinstance(stamp, np.ndarray):
            ax_img.imshow(stamp, cmap="gray")
            add_gap_crosshair(ax_img, gap_frac=0.2, lw=1.0, alpha=0.9)
            ax_img.set_title(self.catalog_thumbnail.get("label", "Finder"), fontsize=11)
            ax_img.set_xticks([])
            ax_img.set_yticks([])
        else:
            ax_img.axis("off")
            ax_img.text(0.5, 0.5, "Finder unavailable", ha="center", va="center", fontsize="small")


        return fig

    
    def _display_name(self) -> str:
        """
        Robust display name for plots.
        Prefer external_id, fall back to internal object id.
        """
        obj = self.r.object
        ext = obj.external_id
        if not ext:
            return str(obj.id)
        ext = str(ext).strip()
        if ext == "" or ext.lower() == "none":
            return str(obj.id)
        return ext


    # ------------------------------------------------------------
    # Plot the Lightcurve
    # ------------------------------------------------------------

    def plot_lightcurve(
        self,
        bands: Optional[list[str]] = None,
        t_lim: Optional[list[float]] = None,
        max_tago: Optional[float] = None,
        sigma_limit: Optional[float] = None,
        ax: Optional[Axes] = None,
        use_mag: bool = True,
        **kwargs
    ) -> Axes:
        """
        Plot photometric light curve.

        bands: Filter bands to plot
        t_lim: Min and max JD time to plot.
        max_tago: Only include t_ago days past now.
        sigma_limit: Only plot detections above this threshold.
        ax: Existing matplotlib axes
        use_mag: Plot magnitudes instead of fluxes
        **kwargs: Additional arguments passed to matplotlib errorbar
        """

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        if self.phot_table is None:
            self.create_table()

        if self.phot_table is None or len(self.phot_table) == 0:
            ax.text(0.5, 0.5, "No photometry", ha="center", va="center")
            ax.set_axis_off()
            return ax

        df = self.phot_table

        if bands is None:
            bands = set(df["band"])

        if t_lim is None:
            t_lim = [df["time"].min(), df["time"].max()]

        if max_tago is not None:
            t_lim[0] = Time.now().jd - max_tago

        if sigma_limit is None:
            sigma_limit = 0.0

        if use_mag:
            # keep only positive fluxes
            pos = df["flux"].astype(float) > 0
            df = df.loc[pos].copy()

            if len(df) == 0:
                ax.text(
                    0.5,
                    0.5,
                    "No positive flux points",
                    ha="center",
                    va="center",
                )
                ax.set_axis_off()
                return ax

            # per-point zp preferred; fallback
            if "zp" not in df.columns:
                df["zp"] = 31.4

            df["mag"] = -2.5 * np.log10(df["flux"].astype(float)) + df["zp"].astype(float)
            df["magerr"] = np.abs(
                -2.5 * df["fluxerr"].astype(float)
                / (df["flux"].astype(float) * np.log(10))
            )

        # Signal-to-noise for filtering
        snr = np.abs(df["flux"].astype(float)) / df["fluxerr"].astype(float)

        for band in bands:
            if band not in BANDINFO:
                print(f"Warning: band {band} not in BANDINFO, skipping")
                continue

            mask = (
                (df["time"] > t_lim[0]) &
                (df["time"] < t_lim[1]) &
                (df["band"] == band) &
                (snr > sigma_limit)
            )

            if not np.any(mask):
                continue

            if use_mag:
                ax.errorbar(
                    df.loc[mask, "time"],
                    df.loc[mask, "mag"],
                    yerr=df.loc[mask, "magerr"],
                    fmt="o",
                    label=BANDINFO[band]["label"],
                    markersize=5,
                    color=BANDINFO[band]["c"],
                    **kwargs,
                )
            else:
                ax.errorbar(
                    df.loc[mask, "time"],
                    df.loc[mask, "flux"],
                    yerr=df.loc[mask, "fluxerr"],
                    fmt="o",
                    label=BANDINFO[band]["label"],
                    markersize=5,
                    color=BANDINFO[band]["c"],
                    **kwargs,
                )

        # Labels & cosmetics
        ax.legend(fontsize="small")

        ax.set_xlabel("Time (JD)")

        if use_mag:
            ax.set_ylabel("Magnitude [AB]")
            ax.invert_yaxis()
        else:
            ax.set_ylabel(f"Flux (zp {self.phot_table['zp'][0]})")

        # robust title (avoid "None (id)")
        name = self._display_name()
        obj_id = str(self.r.object.id)
        z = self.r.host[0].redshift if len(self.r.host) > 0 and self.r.host[0].redshift is not None else None
        z_txt = f"{float(z):.2f}" if z is not None else "?"
        ax.set_title(f"ID: {name} ({obj_id})  z: {z_txt}" if str(name)!=obj_id else f"ID: {name}  z: {z_txt}")

        ax.grid(True, alpha=0.3)

        return ax
    
    # ------------------------------------------------------------
    # Display the report (using methods above)
    # ------------------------------------------------------------
    
    def draw_summary_row(
        self,
        subfig,
        lc_kwargs: dict | None = None,
        cls_threshold: float = 0.01,
        *,
        classifier: int | str = 0,
        model: int = 0,
    ) -> None:
        """
        Draw a summary row into an existing (sub)figure:
        [ lightcurve | classification | host info | finder ].
        """
        lc_kwargs = lc_kwargs or {}

        gs = GridSpec(
            nrows=1,
            ncols=3,
            figure=subfig,
            width_ratios=[2.2, 1.2, 2.0],
            wspace=0.25,
        )

        # Lightcurve
        ax_lc = subfig.add_subplot(gs[0, 0])
        self.plot_lightcurve(ax=ax_lc, **lc_kwargs)

        # Classification
        ax_cls = subfig.add_subplot(gs[0, 1], projection="polar")

        try:
            cls_list = self.r.classification

            if isinstance(classifier, int):
                c = cls_list[classifier]
            else:
                matches = [c for c in cls_list if getattr(c, "name", None) == classifier]
                if not matches:
                    raise ValueError(f"No classifier with name={classifier!r}. Available: {[c.name for c in cls_list]}")
                c = matches[0]

            m = c.models[model]
            probs = {str(k): float(v) for k, v in (m.probabilities or {}).items()}
            title = f"{c.name} (v{c.version}): {m.model}"

            create_classprob_radar(probs, ax_cls, threshold=cls_threshold, title="Classification")

            # explicit label (like before)
            ax_cls.text(
                0.5, -0.12,
                title,
                transform=ax_cls.transAxes,
                ha="center", va="top",
                fontsize=9, alpha=0.75,
            )

        except Exception:
            ax_cls.axis("off")
            ax_cls.text(
                0.5, 0.5,
                "No classification selected/available",
                ha="center", va="center", fontsize="small",
            )
            probs = {}

        host_sub = subfig.add_subfigure(gs[0, 2])

        probs_for_text = probs


        self.show_hostinfo(
            fig=host_sub,
            classprobs=probs_for_text,
            cls_threshold=cls_threshold,
            cls_max_items=12,
            classifier=classifier,
            model=model
        )



    def plot_summary_row(
        self,
        lc_kwargs: dict | None = None,
        cls_threshold: float = 0.01,
        classifier: int | str = 0,
        model: int = 0,
        figsize: tuple[float, float] = (18, 4),
    ):
        """
        Wrapper for `draw_summary_row` that creates a new figure and returns it.
        """
        fig = plt.figure(figsize=figsize, constrained_layout=False)
        self.draw_summary_row(fig, lc_kwargs=lc_kwargs, classifier=classifier, model=model, cls_threshold=cls_threshold)
        return fig


