

from typing import Dict, List, Callable, Optional, Iterable
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time

import math
from matplotlib.gridspec import GridSpec
import pickle
from pathlib import Path

from AmpelReport import AmpelTransientReport



class AmpelReportSet:
    """
    Container and manager for multiple `AmpelTransientReport` objects.

    This class maintains:
    - A unique report per AMPEL object ID (newer reports replace older ones)
    - A dynamic set of filters selecting an active subset of Transients.

    It also provides convenience methods for:
    - Batch visualization (lightcurves, classifications, host info)
    - Grid and row-wise summary plots
    - (Serialization / caching to disk)

    Notes
    -----
    - Filters are applied lazily and recomputed whenever reports or filters change.
    - Time-based filters operate on photometry tables, not derived features.
    """

    def __init__(
        self,
        reports: Optional[Iterable[AmpelTransientReport]] = None,
    ):
        # All reports indexed by object id
        self._reports: Dict[str, AmpelTransientReport] = {}

        # Active filters (callables)
        self._filters: List[Callable[[AmpelTransientReport], bool]] = []

        # Cached active set
        self._active_ids: set[str] = set()

        if reports is not None:
            for r in reports:
                self.add_report(r)

        self._reevaluate_active_set()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self._reports)

    def __iter__(self):
        return iter(self.active_reports)

    @staticmethod
    def _latest_phot_time(report: AmpelTransientReport) -> float:
        """
        Return the latest photometry timestamp in JD.
        Returns -np.inf if none found.
        """
        if report.phot_table is None:
            report.create_table()

        if len(report.phot_table) == 0:
            return -np.inf

        return float(report.phot_table["time"].max())

    def _reevaluate_active_set(self):
        """
        Recompute which reports pass all filters.
        All filters are combined using logical AND.
        If a filter raises an exception for a report, the report is rejected.
        """
        active = set()

        for obj_id, report in self._reports.items():
            passed = True
            for f in self._filters:
                try:
                    if not f(report):
                        passed = False
                        break
                except Exception:
                    passed = False
                    break

            if passed:
                active.add(obj_id)

        self._active_ids = active

    # ------------------------------------------------------------------
    # Report management
    # ------------------------------------------------------------------

    def add_report(self, report: AmpelTransientReport):
        """
        Add a report. If a report for the same object already exists,
        replace it if the new one is newer as determined by the age of the
        latest photometry point.
        """
        obj_id = report.r["object"]["id"]

        if obj_id not in self._reports:
            self._reports[obj_id] = report
        else:
            old = self._reports[obj_id]
            if self._latest_phot_time(report) > self._latest_phot_time(old):
                self._reports[obj_id] = report

        self._reevaluate_active_set()

    def remove_report(self, object_id: str):
        """
        Remove a report by object id.
        """
        if object_id in self._reports:
            del self._reports[object_id]
        self._reevaluate_active_set()

    def print_status(self):
        """
        Print a short summary of the report set state, including
        total number of stored reports and number of active reports.        
        """
        total = len(self._reports)
        active = len(self._active_ids)
        print(f"AmpelReportSet status: {active} active / {total} total reports")

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def clear_filters(self):
        """
        Remove all active filters.
        """
        self._filters = []
        self._reevaluate_active_set()

    def add_filter(self, filter_func: Callable[[AmpelTransientReport], bool]):
        """
        Add a filter function.

        The filter must accept an `AmpelTransientReport` and return a boolean.
        Filters are cumulative and remain active until cleared.        
        Todo: add check such a filter is not already present? Replace if so? Only one of each type?
        """
#        for f in self._filters:
#            if f == filter_func:
#                return
        self._filters.append(filter_func)
        self._reevaluate_active_set()

    # ---- Common filter helpers ---------------------------------------


    def filter_class_prob(self, class_name: str, min_prob: float=0.0, max_prob: float=1.0):
        """
        Keep reports with classification probability within [min_prob, max_prob] for class_name.
        Reject reports with no classifications.
        Will use the maximum probability over all classifiers.
        """
        def _f(r: AmpelTransientReport) -> bool:
            prob = r.get_class_probability( class_name, mode='max' )
            return min_prob <= prob <= max_prob

        self.add_filter(_f)

    def filter_age(self, 
                   min_tago: float = 0, max_tago: float = np.inf, 
                   min_duration: float = 0, max_duration: float = np.inf,
                   sigma_limit: float = 3.0, now_jd: Optional[float] = None
                   ):
        """
        Keep transients within age constraints.

        Definitions
        -----------
        - Detection: a photometric point with |flux / fluxerr| >= sigma_limit
        - Time ago (tago): now_jd - time of last detection (days)
        - Duration: time between first and last detection (days)

        Notes
        -----
        - Times are derived from the photometry table (Julian Date).
        - Reports with no significant detections are rejected.        
        """
        if now_jd is None:
            now_jd = Time.now().jd

        def _f(r: AmpelTransientReport) -> bool:
            if r.phot_table is None:
                r.create_table()
            if len(r.phot_table) == 0:
                return False
            sig_mask = np.abs( r.phot_table["flux"] ) / r.phot_table["fluxerr"] >= sigma_limit
            if not np.any(sig_mask):
                return False
            det_times = r.phot_table["time"][sig_mask]
            t_first = float(np.min(det_times))
            t_last = float(np.max(det_times))
            tago = now_jd - t_last
            duration = t_last - t_first
            return (min_tago <= tago <= max_tago) and (min_duration <= duration <= max_duration)
        
        self.add_filter(_f)

    def filter_sky_region(
            self,
            ra_range: tuple[float, float],
            dec_range: tuple[float, float],
        ):
            """
            Keep objects within an RA/Dec box.
            Todo: support other region shapes, probably through healpix masks.
            RA wrapping (e.g. regions crossing 0 deg) is not supported.
            """

            def _f(r: AmpelTransientReport) -> bool:
                ra = r.r["object"]["ra"]
                dec = r.r["object"]["dec"]
                return (
                    ra_range[0] <= ra <= ra_range[1]
                    and dec_range[0] <= dec <= dec_range[1]
                )

            self.add_filter(_f)

        # ------------------------------------------------------------------
        # Accessors
        # ------------------------------------------------------------------
    @property
    def reports(self) -> List[AmpelTransientReport]:
        """
        All stored reports.
        """
        return list(self._reports.values())

    @property
    def active_reports(self) -> List[AmpelTransientReport]:
        """
        Reports passing current filters.
        """
        return [self._reports[i] for i in self._active_ids]

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------

    def plot_lightcurves(
        self,
        **kwargs,
    ):
        """
        Plot lightcurves of all active reports in separate figures.
        """
        axes = []
        for r in self.active_reports:
            ax = r.plot_lightcurve(**kwargs)
            axes.append(ax)
        return axes

    def show_classifications(self):
        """
        Show classification radar plots for all active reports.
        """
        axes = []
        for r in self.active_reports:
            ax = r.show_classification()
            axes.append(ax)
        return axes

    def show_hosts(self):
        """
        Show host/environment info for all active reports.
        """
        figs = []
        for r in self.active_reports:
            fig = r.show_hostinfo()
            figs.append(fig)
        return figs


    # ==============================================================
    # Batch plotting
    # ==============================================================

    def plot_lightcurves_grid(
        self,
        ncols: int = 3,
        figsize: tuple = (15, 10),
        **kwargs,
    ):
        """
        Plot lightcurves of active reports in a grid.

        Args:
            ncols: Number of columns
            figsize: Figure size
            **kwargs: Passed to AmpelTransientReport.plot_lightcurve

        Returns:
            fig, axes
        """
        reports = self.active_reports
        n = len(reports)

        if n == 0:
            raise RuntimeError("No active reports to plot")

        nrows = math.ceil(n / ncols)

        fig = plt.figure(figsize=figsize)
        gs = GridSpec(nrows, ncols, figure=fig)

        axes = []
        for i, report in enumerate(reports):
            row, col = divmod(i, ncols)
            ax = fig.add_subplot(gs[row, col])
            report.plot_lightcurve(ax=ax, **kwargs)
            axes.append(ax)

        # Hide unused axes
        for j in range(i + 1, nrows * ncols):
            row, col = divmod(j, ncols)
            fig.add_subplot(gs[row, col]).axis("off")

        fig.tight_layout()
        return fig, axes


    def plot_classification_grid(
        self,
        ncols: int = 3,
        figsize: tuple = (15, 10),
    ):
        """
        Plot classification radar plots in a grid.
        """
        reports = self.active_reports
        n = len(reports)

        if n == 0:
            raise RuntimeError("No active reports to plot")

        nrows = math.ceil(n / ncols)

        fig = plt.figure(figsize=figsize)
        gs = GridSpec(nrows, ncols, figure=fig)

        axes = []
        for i, report in enumerate(reports):
            row, col = divmod(i, ncols)
            ax = fig.add_subplot(gs[row, col], projection="polar")
            report.show_classification(ax=ax)
            axes.append(ax)

        for j in range(i + 1, nrows * ncols):
            row, col = divmod(j, ncols)
            fig.add_subplot(gs[row, col]).axis("off")

        fig.tight_layout()
        return fig, axes


    def plot_hostinfo_grid(
        self,
        ncols: int = 2,
        figsize: tuple = (16, 6),
    ):
        """
        Plot host/environment info panels in a grid.
        """
        reports = self.active_reports
        n = len(reports)

        if n == 0:
            raise RuntimeError("No active reports to plot")

        nrows = math.ceil(n / ncols)

        fig = plt.figure(figsize=figsize)
        outer = GridSpec(nrows, ncols, figure=fig)

        for i, report in enumerate(reports):
            row, col = divmod(i, ncols)
            subfig = fig.add_subfigure(outer[row, col])
            report.show_hostinfo(fig=subfig)

        fig.tight_layout()
        return fig


    # ==============================================================
    # Serialization / caching
    # ==============================================================

    def save(self, path: str | Path):
        """
        Serialize the report set to disk.
        - Uses Python pickle so files are not guaranteed to be portable.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(
                {
                    "reports": self._reports,
                    "filters": self._filters,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )


    @classmethod
    def load(cls, path: str | Path):
        """
        Load a cached AmpelReportSet from disk.
        """
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)

        ars = cls()
        ars._reports = data["reports"]
        ars._filters = data["filters"]
        ars._reevaluate_active_set()

        return ars


    @staticmethod
    def cache_exists(path: str | Path) -> bool:
        return Path(path).exists()


    # ==============================================================
    # Row-wise summary plotting
    # ==============================================================

    def plot_summary_rows(
        self,
        figsize: tuple = (18, 4),
        lc_kwargs: Optional[dict] = None,
        max_rows: Optional[int] = None,
    ):
        """
        Plot a summary grid with one row per transient:
        [ lightcurve | classification | host info ]

        Args:
            figsize: Base figure size per row (width, height)
            lc_kwargs: Passed to plot_lightcurve
            max_rows: Optional cap on number of rows

        Notes
        - Figure height scales linearly with the number of rows.
        - Host panels use subfigures to allow mixed text/image layouts.

        Returns:
            fig, axes_dict
        """
        lc_kwargs = lc_kwargs or {}

        reports = self.active_reports
        if max_rows is not None:
            reports = reports[:max_rows]

        nrows = len(reports)
        if nrows == 0:
            raise RuntimeError("No active reports to plot")

        fig = plt.figure(
            figsize=(figsize[0], figsize[1] * nrows),
            constrained_layout=False,
        )

        # Outer grid: rows x 3 columns
        gs = GridSpec(
            nrows,
            3,
            figure=fig,
            width_ratios=[2.2, 1.2, 2.0],
            wspace=0.25,
            hspace=0.35,
        )

        axes = {}

        for i, report in enumerate(reports):
            obj_id = report.r["object"]["id"]
            axes[obj_id] = {}

            # --------------------------------------------------
            # Lightcurve
            # --------------------------------------------------
            ax_lc = fig.add_subplot(gs[i, 0])
            report.plot_lightcurve(ax=ax_lc, **lc_kwargs)
            ax_lc.set_title(
                f"{report.r['object']['external_id']} ({obj_id})",
                fontsize=10,
            )
            axes[obj_id]["lightcurve"] = ax_lc

            # --------------------------------------------------
            # Classification
            # --------------------------------------------------
            ax_cls = fig.add_subplot(gs[i, 1], projection="polar")
            report.show_classification(ax=ax_cls)
            axes[obj_id]["classification"] = ax_cls

            # --------------------------------------------------
            # Host info (needs its own sub-layout)
            # --------------------------------------------------
            subfig = fig.add_subfigure(gs[i, 2])
            report.show_hostinfo(fig=subfig)
            axes[obj_id]["host"] = subfig

     #   fig.tight_layout()
        return fig, axes





