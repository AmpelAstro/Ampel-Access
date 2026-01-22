#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File:                AmpelReport.py
# License:             BSD-3-Clause
# Author:              jno
# Date:                19.1.2026
# Last Modified Date:  19.1.2026
# Last Modified By:    jno

"""
AmpelReport
===========

Utilities for visualizing and summarizing AMPEL alert reports.

This module provides:
- Lightcurve plotting
- Classification probability visualization
- Host/environment information display

The main entry point is the `AmpelTransientReport` class, which wraps
an `LSSTReport` object from ampel-hu-astro.

Dependencies
------------
numpy, pandas, matplotlib, astropy, pillow, ztfquery, ampel-hu-astro

Notes
-----
Some helper functions are currently duplicated from AMPEL T3 PlotTransientLightcurves.
LSSTReport should be imported without requiring the full AMPEL stack.
"""


from typing import Optional
import io
import requests

import numpy as np
import pandas as pd

from astropy.time import Time

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.pyplot import Figure
from matplotlib.gridspec import GridSpec
from PIL import Image

### These are all methods from Ampel (mainly hu-astro).
# Should prob move to an easy to import package. 

# Models loaded from ampel-hu-astro
from ampel.contrib.hu.model.LSSTReport import (
    Classification,
    Feature,
    Host,
    LSSTReport,
    Object,
    PhotometricPoint,
    LSSTReport
)

# Method largely grabbed from T3 PlotLightcurves 
def get_finder_stamp(
    ra: float,
    dec: float,
    size: int = 240,
    surveys: list[str] | None = None,
    timeout: float = 8.0,
    fov_arcsec: float = 8,
):
    """
    RA, Dec in degrees.
    Size in pixels (square).

    Best-effort finder stamp:
    1) PS1 via get_ps_stamp (returns PIL image)
    2) HiPS/hips2fits fallback as FITS -> we apply our own stretch
    Returns (image_array, label) where image_array is 2D float array, or (None, None).

    Returns
    -------
    image : np.ndarray | None
        2D image or None on failure.
    label : str | None
        Survey label, or None on failure.
    """


    if surveys is None:
        surveys = [
            "CDS/P/Skymapper-color-IRG",
            "CDS/P/DECaLS/DR5/color",
            "CDS/P/DSS2/color",
        ]

    LABELS = {
        "CDS/P/Skymapper-color-IRG": "SkyMapper",
        "CDS/P/DECaLS/DR5/color": "DECaLS",
        "CDS/P/DSS2/color": "DSS2",
    }

    fov_deg = fov_arcsec / 3600.0

    for hips in surveys:
        print('look for', hips)
        try:
            r = requests.get(
                "https://alasky.cds.unistra.fr/hips-image-services/hips2fits",
                params={
                    "hips": hips,
                    "ra": ra,
                    "dec": dec,
                    "fov": fov_deg,
                    "width": size,
                    "height": size,
                    "format": "png",
                },
                timeout=timeout,
            )
            if r.status_code != 200 or not r.content:
                continue

            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            arr = np.asarray(img).astype(float)

            # RGB → Graustufen (optional, aber konsistent)
            arr = (
                0.2126 * arr[..., 0]
                + 0.7152 * arr[..., 1]
                + 0.0722 * arr[..., 2]
            )
            arr /= 255.0

            return arr, LABELS.get(hips, hips)

        except Exception:
            continue

    return None, None

# Method largely grabbed from T3 PlotLightcurves 
def normalize_image_for_imshow(arr: np.ndarray) -> np.ndarray:
    """
    Convert possible image cubes into a 2D (H,W) array suitable for imshow.
    Handles shapes:
      (H,W) -> unchanged
      (H,W,3/4) -> luminance
      (3/4,H,W) -> luminance
      (N,H,W) with N!=3/4 -> take first plane
    """
    a = np.asarray(arr)

    # Already 2D
    if a.ndim == 2:
        return a

    # Channel-last RGB/RGBA
    if a.ndim == 3 and a.shape[2] in (3, 4):
        rgb = a[..., :3].astype(float)
        return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]

    # Channel-first RGB/RGBA
    if a.ndim == 3 and a.shape[0] in (3, 4):
        rgb = a[:3, ...].astype(float)
        return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]

    # Any other cube -> take first plane
    if a.ndim == 3:
        return a[0, ...] if a.shape[0] < a.shape[-1] else a[..., 0]

    raise TypeError(f"Unsupported stamp shape for imshow: {a.shape}")

# Method largely grabbed from T3 PlotLightcurves 
def create_classprob_radar(
        classprobs: dict[str, float],
        ax: Axes,
        threshold: float = 0.01,
        title: str = "Class probabilities",
    ):
        """
        Radar chart for class probabilities.
        Shows all classes with p >= threshold.
        Rings at 0.25, 0.5, 0.75, 1.0.
        Gracefully handles N=1/2 via bar plot fallback.
        """
        # Filter + sort (descending)
        items = [(k, float(v)) for k, v in classprobs.items() if float(v) >= threshold]
        items.sort(key=lambda kv: kv[1], reverse=True)

        if not items:
            ax.axis("off")
            ax.text(0.5, 0.5, "No classes >= {:.2f}".format(threshold),
                    ha="center", va="center", fontsize="small")
            return ax 

        labels = [k for k, _ in items]
        vals = np.array([v for _, v in items], dtype=float)

        ax.set_title(title, fontsize="small", pad=8)
        ax.set_ylim(0.0, 1.0)

        # Rings / radial ticks
        rticks = [0.5, 1.0]
        ax.set_yticks(rticks)
        ax.set_yticklabels([str(x) for x in rticks], fontsize=6, alpha=0.6)
        ax.tick_params(axis="y", pad=2)

        # If too few points, a true polygon radar looks bad → fallback to bars in polar
        n = len(vals)
        if n < 3:
            theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
            width = (2 * np.pi) / max(n, 1) * 0.8
            ax.bar(theta, vals, width=width, alpha=0.35, edgecolor="black", linewidth=0.6)
            ax.set_xticks(theta)
            ax.set_xticklabels(labels, fontsize=7)
            ax.set_theta_offset(np.pi / 2.0)
            ax.set_theta_direction(-1)
            ax.grid(True, alpha=0.4)
            return ax 

        # Standard radar polygon
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        theta_closed = np.r_[theta, theta[0]]
        vals_closed = np.r_[vals, vals[0]]

        ax.set_theta_offset(np.pi / 2.0)
        ax.set_theta_direction(-1)

        ax.plot(theta_closed, vals_closed, linewidth=1.2)
        ax.fill(theta_closed, vals_closed, alpha=0.25)

        ax.set_xticks(theta)
        ax.set_xticklabels(labels, fontsize=7)
        ax.grid(True, alpha=0.4)

        return ax 

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

    Parameters
    ----------
    report : LSSTReport
        Parsed AMPEL alert report containing object metadata, photometry,
        classifications, and host information.
    """

    r: LSSTReport
    phot_table: None
    catalog_thumbnail = {'stamp':None, 'label':None}

    def __init__(self, report: LSSTReport):
        self.r = report
        self.phot_table = None

    def get_class_probability( self, class_name: str, mode: str='max' ) -> float:
        """
        Get class probability for given class name.
        If multiple classifications exist, return max, min or average probability.
        Args:
            class_name: Name of class to retrieve
            mode: 'max', 'min' or 'avg' to select how to combine multiple probabilities
        Returns:
            Class probability (float) or -1.0 if class not found

        """
        probs = []
        for classification in self.r['classification']:
            for model in classification['models']:
                if class_name in model['probabilities']:
                    probs.append( model['probabilities'][class_name] )
        
        if len(probs)==0:
            return -1.0
        if mode=='max':
            return max(probs)
        elif mode=='min':
            return min(probs)
        elif mode=='avg':
            return sum(probs)/len(probs)
        else:
            raise ValueError('Invalid mode: {}'.format(mode))   


    def create_table(self, backup_zp=25.0):
        """
        Convert existing list of PhotometricPoints to pandas table.

        Assumes photometry datapoints contain:
        - time (JD)
        - flux
        - fluxerr
        - zp
        - band
        """

        if len(self.r['photometry'])==0:
            self.phot_table = None
            return

        self.phot_table = pd.DataFrame.from_dict( self.r['photometry'] )

        # If phot points use different zeropoints, shift to backup value.
        if len(set(self.phot_table["zp"]))>1:
            scaling = 10**( 2.5* ( backup_zp - self.phot_table["zp"] ) )
            self.phot_table['flux'] *= scaling
            self.phot_table['fluxerr'] *= scaling
            self.phot_table['zp'] = backup_zp

    
    def get_catalogimage(self, surveys: Optional[list[str]] = None):
        """
        Load a catalog thumbnail from CDS.
        """
        if surveys is None:
            surveys = ["CDS/P/DSS2/color"]

        stamp, stamp_label = get_finder_stamp(
            ra=self.r['object']['ra'], dec=self.r['object']['dec'], 
            size=240, fov_arcsec=180.0, surveys = surveys
        )
        if stamp is None:
            # Set to -1 to indicate we tried but failed
            self.catalog_thumbnail = {'stamp':-1, 'label':'fail'}
        else:
            self.catalog_thumbnail = {
                'stamp':normalize_image_for_imshow(stamp), 
                'label':stamp_label
            }

    def show_classification(
        self, 
        ax: Optional[Axes] = None, 
    ) -> Axes:
        """
        Show available classification.
        
        Args:
            fig: Existing matplotlib figure (creates new if None)
        Returns:
            matplotlib axis object with the plot
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='polar')

        if len(self.r['classification'])==0:
            print('No classifications available')
            return
        if len(self.r['classification'])>1:
            print('Multiple classifications, currently only showing first')

        return create_classprob_radar(
            self.r['classification'][0]['models'][0]['probabilities'],
            ax,
            threshold = 0.01,
            title = '{}: {}'.format(self.r['classification'][0]['name'],self.r['classification'][0]['models'][0]['model']),
        )

    
    def show_hostinfo(
        self, 
        fig: Optional[Figure] = None, 
    ) -> Figure:
        """
        Show available host / environment information.
        
        Args:
            fig: Existing matplotlib figure (creates new if None)
        Returns:
            matplotlib figure object with the plot
        """
        
        if fig is None:
            fig = plt.figure(figsize=(8, 5))
            gs = GridSpec(1, 2, figure=fig, width_ratios=[1.2, 1])
            ax_text = fig.add_subplot(gs[0])
            ax_img = fig.add_subplot(gs[1])
        else:
            # So, this will grab the full figureß
            gs = GridSpec(1, 2, figure=fig, width_ratios=[1.2, 1])
            ax_text = fig.add_subplot(gs[0])
            ax_img = fig.add_subplot(gs[1])

        # Ensure catalog thumbnail exists
        if self.catalog_thumbnail["stamp"] is None:
            self.get_catalogimage()

        # ---- TEXT PANEL ----
        ax_text.axis("off")

        # Collect information to print
        lines = []
        if len(self.r['host'])>0:
            for key in ['redshift', 'distance', 'info']:
                if (val:=(self.r['host'][0].get(key,None))) is not None:
                    lines.append(f"{key}: {val}")

        if len(lines)>0:
            ax_text.text(
                0.0,
                1.0,
                "\n".join(lines),
                va="top",
                ha="left",
                fontsize=10,
                family="monospace",
                transform=ax_text.transAxes,
            )    
            ax_text.set_title("Host / Environment Info", fontsize=11, pad=10)
    
        # ---- IMAGE PANEL ----
        if not self.catalog_thumbnail["label"] == 'fail':        
            im = ax_img.imshow(self.catalog_thumbnail["stamp"], origin="lower", cmap="gray")
            ax_img.set_title(self.catalog_thumbnail["label"], fontsize=11)
            ax_img.set_xticks([])
            ax_img.set_yticks([])
    
        return fig

    def plot_lightcurve(
        self,
        bands: Optional[list[str]] = None,
        t_lim: Optional[list[float]] = None,
        max_tago: Optional[float] = None,
        sigma_limit: Optional[float] = None,
        ax: Optional[Axes] = None,
        **kwargs
    ) -> Axes:
        """
        Plot photometric light curve.
        
        Args:
            bands: Filter bands to plot (None for all bands)
            t_lim: Min and max JD time to plot. (None for all time)
            max_tago: Only include t_ago days past now. (None for all time)
            sigma_limit: Only plot detections above this threshold.
            ax: Existing matplotlib axes (creates new if None)
            **kwargs: Additional arguments passed to matplotlib errorbar
            
        Returns:
            matplotlib axes object with the plot
        """
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        if self.phot_table is None:
            self.create_table()

        # Setup limits 
        if bands is None:
            bands = set( self.phot_table["band"] )
        if t_lim is None:
            t_lim = [self.phot_table["time"].min(), self.phot_table["time"].max()]
        if max_tago is not None:
            t_lim[0] = Time.now().jd-max_tago
        if sigma_limit is None:
            sigma_limit = 0
            
        # Plot
        for band in bands:
            if not band in BANDINFO:
                print('Warning: band {} not in BANDINFO, skipping'.format(band))
                continue
            mask = ( 
                (self.phot_table["time"]>t_lim[0]) & 
                (self.phot_table["time"]<t_lim[1]) & 
                (self.phot_table["band"]==band) & 
                (np.abs(self.phot_table["flux"]) / self.phot_table["fluxerr"]) > sigma_limit
            )
            ax.errorbar(
                self.phot_table.loc[mask,"time"], 
                self.phot_table.loc[mask,"flux"], 
                yerr=self.phot_table.loc[mask,"fluxerr"], 
                fmt='o', label=BANDINFO[band]['label'],
                markersize=5, color=BANDINFO[band]['c'], **kwargs
            )

        ax.legend()
        
        ax.set_xlabel('Time (JD)')
        ax.set_ylabel('Flux (zp {})'.format(self.phot_table["zp"][0]))
        ax.set_title( '{} - AMPEL {}'.format(
            self.r['object']['external_id'], self.r['object']['id'], 
        ) )
        ax.grid(True, alpha=0.3)
        
        return ax
