#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File:                AmpelReportSet.py
# License:             BSD-3-Clause
# Author:              jno
# Date:                19.1.2026
# Last Modified Date:  19.2.2026
# Last Modified By:    Felix Fischer

""" 
AmpelReportSet: 
===========

Utilities for visualizing and summarizing a set of AMPEL alert reports.

This module defines the `AmpelReportSet` class. It provides functionality for:
- Streaming updates from Hopskotch topics with incremental offsets.
- Merging new reports with existing ones based on object ID and photometry time into a cache of per-object pickles.
- Summarizing classifier/model availability across the set of reports.
- Dynamic filtering of the active report subset based on user-defined criteria (e.g. class probabilities, redshift, detection count).
- Checking for potential LSST sources in IceCube's HEALPix skymap.
- Producing summary figures showing light curves, classifications, host info, and finder stamps of each active report.

"""



from __future__ import annotations

from typing import Dict, List, Callable, Optional, Iterable, Any
import json
import math
import pickle
from pathlib import Path
from collections import Counter, defaultdict
import os
import requests

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import QTable
from matplotlib.gridspec import GridSpec

from hop import Stream
from hop.io import StartPosition

import astropy_healpix as ah
from icecube_tools import select_pixels_sets
import ligo.skymap.plot
from ligo.skymap.io.fits import read_sky_map

from .AmpelReport import AmpelTransientReport




class AmpelReportSet:

    def __init__(
        self,
        reports: Optional[Iterable[AmpelTransientReport]] = None,
    ):
        """
        For AmpelReport objects, initializes the internal state dictionaries and sets the cache/sync metadata.
        """
        self._reports: Dict[str, AmpelTransientReport] = {}
        self._filters: List[Callable[[AmpelTransientReport], bool]] = []
        self._filter_labels: List[str] = []
        self._active_ids: set[str] = set()

        # cache/sync metadata
        self._cache_root: Optional[Path] = None
        self._topic_url: Optional[str] = None
        self._group_id: Optional[str] = None

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
        Return latest photometry time (JD). -inf if none.
        """
        if report.phot_table is None:
            report.create_table()
        if report.phot_table is None or len(report.phot_table) == 0:
            return -np.inf
        return float(report.phot_table["time"].max())

    def _reevaluate_active_set(self) -> None:
        """
        Re-evaluate active set of object IDs based on the current set of filters.
        If all filters pass for a report, add the object ID to the active set.
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

    def add_report(self, report: AmpelTransientReport) -> None:
        """
        Add/replace report for an object id.
        If the object id is already present, replace is it if the given is newer.
        Latest photopoint time is used for comparison.
        """
        obj_id = str(report.r.object.id)

        if obj_id not in self._reports:
            self._reports[obj_id] = report
        else:
            old = self._reports[obj_id]
            if self._latest_phot_time(report) > self._latest_phot_time(old):
                self._reports[obj_id] = report

        self._reevaluate_active_set()

    def remove_report(self, object_id: str) -> None:
        """
        Remove the report with the given object_id from the set of AmpelReports.
        If the report is not present, do nothing.
        """
        if object_id in self._reports:
            del self._reports[object_id]
        self._reevaluate_active_set()

    def print_status(self) -> None:
        """
        Print status of the set of Ampel Reports.
        Prints the number of active reports vs the total number of reports.
        """
        total = len(self._reports)
        active = len(self._active_ids)
        print(f"AmpelReportSet status: {active} active / {total} total reports")

    # ------------------------------------------------------------------
    # Merge helpers (static/class methods)
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_photometry(old_list: list[dict], new_list: list[dict]) -> list[dict]:
        """
        Merge two lists of photometry dictionaries and deduplicate so that newer photometry wins.
        """
        if not old_list:
            old_list = []
        if not new_list:
            new_list = []

        df_old = pd.DataFrame(old_list)
        df_new = pd.DataFrame(new_list)

        # Ordering for deduplication: everything from new is "newer" than everything from old
        # and within new, the order of the lists matters.
        # The "_seq" column is used to break ties and refering to the "streaming time" order.
        if len(df_old) > 0:
            df_old["_seq"] = np.arange(len(df_old), dtype=np.int64)
        if len(df_new) > 0:
            df_new["_seq"] = np.arange(len(df_new), dtype=np.int64) + 10_000_000  # surely bigger than old
        
        df = pd.concat([df_old, df_new], ignore_index=True)


        df["time"] = pd.to_numeric(df.get("time"), errors="coerce").round(6)
        df["band"] = df.get("band").astype(str)

        # Drop junk (NaN in time/band)
        df = df.dropna(subset=["time", "band"])

        # First sort by time, then band, then _seq
        df = df.sort_values(["time", "band", "_seq"])
        # And then drop duplicates and for same (time,band) keep the latest (= highest _seq)
        df = df.drop_duplicates(subset=["time", "band"], keep="last")

        # And finally sort by time and drop _seq as its not needed anymore
        df = df.sort_values("time").drop(columns=["_seq"], errors="ignore")
        return df.to_dict(orient="records")


    @staticmethod
    def _prefer_new_block(old: Any, new: Any) -> Any:
        """
        Latest wins, but don't drop old block if new is None/empty.
        """
        # Check for None/empty
        if new is None:
            return old
        
        # Check for empty list/dict
        if isinstance(new, list) and len(new) == 0:
            return old
        if isinstance(new, dict) and len(new) == 0:
            return old
        
        # Otherwise prefer new
        return new

    @classmethod
    def _merge_report_dict(cls, old_r: dict, new_r: dict) -> dict:
        """
        Merge two LSSTReport-like dicts.
        - photometry: union/dedup
        - classification/features/host/object/state: latest wins (but keep old if new missing/empty)
        """
        out = dict(old_r)  # shallow copy

        # Merge photometry
        out["photometry"] = cls._merge_photometry(
            old_r.get("photometry", []),
            new_r.get("photometry", []),
        )

        # Merge other blocks
        for k in ["classification", "features", "host", "object", "state"]:
            out[k] = cls._prefer_new_block(old_r.get(k), new_r.get(k))

        return out

    # ------------------------------------------------------------------
    # Cache IO (per-object pickle + meta.json)
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_cache_layout(cache_root: Path) -> tuple[Path, Path]:
        """
        Check for cache directory structure, otherswise create it. Returns (obj_dir, meta_path).
        """
        cache_root = Path(cache_root)
        obj_dir = cache_root / "objects"
        obj_dir.mkdir(parents=True, exist_ok=True)
        meta_path = cache_root / "meta.json"
        return obj_dir, meta_path

    @staticmethod
    def _utc_now_iso() -> str:
        """
        Return current UTC time in ISO 8601 format with 'Z' suffix
        """
        return pd.Timestamp.utcnow().isoformat() + "Z"

    @classmethod
    def load_cache(cls, cache_root: str | Path) -> "AmpelReportSet":
        """
        Load per-object pickles from cache_root/objects/*.pkl.
        """
        
        cache_root = Path(cache_root)
        obj_dir, meta_path = cls._ensure_cache_layout(cache_root)

        ars = cls()
        ars._cache_root = cache_root

        # Load meta.json if exists (to get topic_url, group_id). Optional.
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                ars._topic_url = meta.get("topic_url")
                ars._group_id = meta.get("group_id")
            except Exception:
                pass

        # Load individual object pickles 
        for pkl in obj_dir.glob("*.pkl"):
            try:
                with open(pkl, "rb") as f:
                    report_dict = pickle.load(f)
                # reading the ID from the report dict. Unusual semantics because it's dict not pedantic.
                obj_id = str(report_dict["object"]["id"])
                ars._reports[obj_id] = AmpelTransientReport(report_dict)
            except Exception:
                # ignore corrupt files rather than killing notebook startup
                continue

        ars._reevaluate_active_set()
        return ars

    def save_cache(self, cache_root: str | Path | None = None) -> None:
        """
        Save each object as objects/<objid>.pkl and write meta.json.
        """
        # ensure cache_root is set
        if cache_root is None:
            if self._cache_root is None:
                raise ValueError("cache_root not set. Provide cache_root or call load_cache() first.")
            cache_root = self._cache_root

        cache_root = Path(cache_root)
        obj_dir, meta_path = self._ensure_cache_layout(cache_root)

        # Save individual object pickles
        for obj_id, rep in self._reports.items():
            with open(obj_dir / f"{obj_id}.pkl", "wb") as f:
                pickle.dump(rep.r.model_dump(), f, protocol=pickle.HIGHEST_PROTOCOL)

        # Save meta.json
        meta = {
            "topic_url": self._topic_url,
            "group_id": self._group_id,
            "last_sync_utc": self._utc_now_iso(),
            "n_objects": len(self._reports),
            "object_ids": sorted(list(self._reports.keys())),
        }
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))

        self._cache_root = cache_root

    # ------------------------------------------------------------------
    # Hopskotch / hop-client sync
    # ------------------------------------------------------------------

    def update_from_stream(
        self,
        topic_url: str,
        group_id: str,
        cache_root: str | Path | None = None,
        start_at_if_new: str = "EARLIEST",
        max_messages: Optional[int] = None,
        persist: bool = True,
    ) -> dict:
        """
        Read messages from Hopskotch and merge into this set.

        Notes:
        - Uses Kafka consumer group offsets via `group_id` for incremental reads.
        - `start_at_if_new` only matters when the group has no committed offsets yet.
        - This uses hop-client's blocking iterator; stopping is either EOS, max_messages,
          or manual interrupt.

        Returns stats dict.
        """
        # ensure cache_root is set
        if cache_root is not None:
            self._cache_root = Path(cache_root)

        # store topic_url and group_id. group_id is needed for setting the starting point of the stream: "Read only what is new since last sync".
        # group_id shows, who you are as a consumer. It is stored in the kafka server side, so we don't need to have it locally.
        self._topic_url = topic_url
        self._group_id = group_id

        #try:
        #    from hop import Stream  # type: ignore
        #    from hop.io import StartPosition  # type: ignore
        #except Exception as e:
        #    raise RuntimeError(
        #        "hop-client is not available in this environment. Install hop-client to read Hopskotch topics."
        #    ) from e

        # Determine start position. Only used if no committed offsets for group_id. 
        # Should be earliest for most use cases. But maybe someone wants to use latest for some reason.
        sp = start_at_if_new.strip().upper()
        if sp == "EARLIEST":
            start_pos = StartPosition.EARLIEST
        elif sp == "LATEST":
            start_pos = StartPosition.LATEST
        else:
            raise ValueError("start_at_if_new must be 'EARLIEST' or 'LATEST'")

        # Create the actual stream.
        stream = Stream(start_at=start_pos, until_eos=True)

        # Counting messages to stop at max_messages and having stats.
        n_messages = 0
        n_added = 0
        n_updated = 0

        # Read message starting from last committed offset for group_id.
        with stream.open(topic_url, "r", group_id=group_id) as s:
            for msg in s:
                new_r = msg.content  # expecting LSSTReport-like dict
                obj_id = str(new_r["object"]["id"])

                # If we already have a report for this object, merge it.
                if obj_id in self._reports:
                    old_r = self._reports[obj_id].r.model_dump()
                    merged = self._merge_report_dict(old_r, new_r)
                    self._reports[obj_id] = AmpelTransientReport(merged)
                    n_updated += 1
                
                # If new object, deduplicate (sometimes artifacts) and add it.
                else:
                    sanitized = dict(new_r)
                    sanitized["photometry"] = self._merge_photometry([], new_r.get("photometry", []))
                    self._reports[obj_id] = AmpelTransientReport(sanitized)
                    n_added += 1

                n_messages += 1
                if max_messages is not None and n_messages >= max_messages:
                    break

        # Re-evaluate active set after updates
        self._reevaluate_active_set()

        # Maybe someone does not want to save after the update, so persist is optional.
        if persist:
            if self._cache_root is None:
                raise ValueError("persist=True but no cache_root set. Provide cache_root or call load_cache().")
            self.save_cache(self._cache_root)

        # Return stats
        return {
            "n_messages": n_messages,
            "n_objects_added": n_added,
            "n_objects_updated": n_updated,
            "n_objects_total": len(self._reports),
        }

    # ------------------------------------------------------------------
    # Classifiers
    # ------------------------------------------------------------------

    def summarize_classifiers(
        self,
        *,
        active_only: bool = True,
        model_index: int = 0,
        cls_threshold: float | None = None,
    ) -> dict[str, Any]:
        """
        Aggregate classification information across reports.

        active_only : If True, only active reports, else all reports.
        model_index : Which model index to inspect within each classifier entry.
            If out of range for a report, that report counts as "no model".
        cls_threshold : If set, only count class labels with p >= threshold when collecting keys.
        """
        reports = self.active_reports if active_only else list(self.reports.values())

        n_total = len(reports)
        n_has_cls = 0
        n_has_any_model = 0

        # counts per classifier identity
        # key: (classifier_name, version, model_name)
        model_counter: Counter[tuple[str, str, str]] = Counter()

        # For each model key: how many reports had it, and how many had probs keys
        n_reports_with_probs: Counter[tuple[str, str, str]] = Counter()
        classkey_union: dict[tuple[str, str, str], set[str]] = defaultdict(set)

        for r in reports:
            cls_list = getattr(r.r, "classification", None) if hasattr(r, "r") else getattr(r, "classification", None)
            if not cls_list:
                continue

            n_has_cls += 1

            for c in cls_list:
                cname = str(getattr(c, "name", ""))
                cver = str(getattr(c, "version", ""))
                models = getattr(c, "models", None) or []
                if len(models) == 0:
                    continue

                # we count "has any model" once per report if at least one classifier has a model
                n_has_any_model = n_has_any_model + 1 if False else n_has_any_model  # placeholder (fixed below)

                if model_index < 0 or model_index >= len(models):
                    # classifier exists but requested model index absent -> just record "missing model"
                    key = (cname, cver, f"<no model idx {model_index}>")
                    model_counter[key] += 1
                    continue

                m = models[model_index]
                mname = str(getattr(m, "model", ""))
                key = (cname, cver, mname)

                model_counter[key] += 1

                probs = getattr(m, "probabilities", None) or {}
                if probs:
                    n_reports_with_probs[key] += 1
                    for k, v in probs.items():
                        try:
                            fv = float(v)
                        except Exception:
                            continue
                        if cls_threshold is not None and fv < float(cls_threshold):
                            continue
                        classkey_union[key].add(str(k))

            if any((getattr(c, "models", None) or []) for c in cls_list):
                n_has_any_model += 1

        # build sorted list for presentation
        per_model = []
        for key, cnt in model_counter.most_common():
            cname, cver, mname = key
            per_model.append(
                {
                    "classifier": cname,
                    "version": cver,
                    "model": mname,
                    "n_reports": int(cnt),
                    "n_reports_with_probs": int(n_reports_with_probs.get(key, 0)),
                    "n_unique_classes": int(len(classkey_union.get(key, set()))),
                    "classes": sorted(classkey_union.get(key, set())),
                }
            )

        return {
            "active_only": active_only,
            "n_total_reports": int(n_total),
            "n_reports_with_classification": int(n_has_cls),
            "n_reports_with_any_model": int(n_has_any_model),
            "per_model": per_model,
        }


    def print_classifiers_summary(
        self,
        *,
        active_only: bool = True,
        model_index: int = 0,
        cls_threshold: float | None = None,
        max_models: int = 20,
        max_classes: int = 30,
    ) -> dict[str, Any]:
        """
        Pretty-print a classifier summary and return the same dict.
        """
        s = self.summarize_classifiers(
            active_only=active_only,
            model_index=model_index,
            cls_threshold=cls_threshold,
        )

        scope = "active_reports" if active_only else "all reports"
        print(f"Classifier summary ({scope}):")
        print(f"  n_total:                 {s['n_total_reports']}")
        print(f"  with classification:     {s['n_reports_with_classification']}")
        print(f"  with >=1 model:          {s['n_reports_with_any_model']}")
        if cls_threshold is not None:
            print(f"  class-key threshold:     p >= {cls_threshold:g}")
        print(f"  inspected model_index:   {model_index}")

        per_model = s["per_model"]
        if not per_model:
            print("\n(no classifier/model occurrences)")
            return s

        print("\nTop classifier/model occurrences:")
        for i, item in enumerate(per_model[:max_models]):
            tail = ""
            classes = item["classes"][:max_classes]
            if item["n_unique_classes"] > len(classes):
                tail = f" (+{item['n_unique_classes'] - len(classes)} more)"
            print(
                f"\n[{i}] {item['classifier']} v{item['version']} / model={item['model']}"
                f"\n    reports: {item['n_reports']}   with_probs: {item['n_reports_with_probs']}"
                f"\n    unique classes: {item['n_unique_classes']}"
            )
            if classes:
                print("    " + ", ".join(classes) + tail)

        return s




    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def clear_filters(self) -> None:
        """
        Clear all filters and reset active set to all reports.
        """
        self._filters = []
        self._filter_labels = []
        self._reevaluate_active_set()

    def add_filter(
        self,
        filter_func: Callable[[AmpelTransientReport], bool],
        *,
        label: str | None = None,
    ) -> None:
        """
        Add a filter function that takes a report and returns True/False. Only reports passing all filters are active.
        """
        filter_func._label = label or filter_func.__name__
        self._filters.append(filter_func)
        self._filter_labels.append(str(label))
        self._reevaluate_active_set()
    
    @property
    def active_filter_config(self) -> str:
        """
        Human-readable one-liner of currently active filters.
        """
        if not self._filter_labels:
            return "Filters: (none)"
        return "Filters: " + " | ".join(self._filter_labels)



    def filter_class_prob(
        self,
        class_name: str,
        min_prob: float = 0.0,
        max_prob: float = 1.0,
        *,
        classifier: int | str = 0,
        model: int = 0,
    ) -> None:
        """
        Filter on class probability.
        """
        # accept both "SNIa" and "P(SNIa)"
        key_raw = str(class_name)

        def _f(r: AmpelTransientReport) -> bool:

            # selected classifier/model path (no helper; use existing get_probabilities)
            try:
                probs = r.get_class_probabilities(classifier=classifier, model=model)
            except Exception:
                return False

            if key_raw in probs:
                prob = float(probs[key_raw])
            else:
                return False

            return (min_prob <= prob <= max_prob)
        
        label = f"{key_raw} ∈ [{min_prob:g}, {max_prob:g}]"

        self.add_filter(_f, label=label)


    def filter_age(
        self,
        min_duration: float = 0.0,
        max_duration: float = np.inf,
        sigma_limit: float = 5.0,
    ) -> None:
        """
        Filter on detection duration only:
        (last_detection_time - first_detection_time) in days.
        """
        def _f(r: AmpelTransientReport) -> bool:
            if r.phot_table is None:
                r.create_table()
            pt = r.phot_table
            if pt is None or len(pt) == 0:
                return False

            for col in ("time", "flux", "fluxerr"):
                if col not in pt:
                    return False

            snr = np.abs(pt["flux"]) / pt["fluxerr"]
            mask = snr >= sigma_limit
            if not np.any(mask):
                return False

            times = pt["time"][mask]
            duration = float(times.max() - times.min())
            return (min_duration <= duration <= max_duration)

        self.add_filter(
            _f,
            label=f"det duration ∈ [{min_duration:g}, {max_duration:g}] d (σ≥{sigma_limit:g})",
        )

    def filter_sky_region(self, ra_range: tuple[float, float], dec_range: tuple[float, float]) -> None:
        """
        Filter reports based on sky region (RA, DEC).
        """
        def _f(r: AmpelTransientReport) -> bool:
            ra = float(r.r.object.ra)
            dec = float(r.r.object.dec)
            return (ra_range[0] <= ra <= ra_range[1]) and (dec_range[0] <= dec <= dec_range[1])

        self.add_filter(
            _f,
            label=f"sky RA∈[{ra_range[0]:g},{ra_range[1]:g}] DEC∈[{dec_range[0]:g},{dec_range[1]:g}]",
        )

    def filter_redshift(self, min_z: float = -np.inf, max_z: float = np.inf) -> None:
        """
        Host redshift filter using host[0]['redshift'].
        """
        def _f(r: AmpelTransientReport) -> bool:
            if len(r.r.host) == 0:
                return False
            z = r.r.host[0].redshift
            if z is None:
                return False
            try:
                zf = float(z)
            except Exception:
                return False
            return min_z <= zf <= max_z


        self.add_filter(_f, label=f"host z ∈ [{min_z:g}, {max_z:g}]")

    def filter_n_det(
        self,
        min_det: int = 1,
        max_det: int = 10**9,
        sigma_limit: float = 5.0,
        band: str | None = None,
    ) -> None:
        """
        Count number of detections defined as |flux|/fluxerr >= sigma_limit.
        Optionally restrict to one band (exact string match).
        """
        def _f(r: AmpelTransientReport) -> bool:
            if r.phot_table is None:
                r.create_table()
            pt = r.phot_table
            if pt is None or len(pt) == 0:
                return False

            # Required columns
            for col in ("time", "flux", "fluxerr"):
                if col not in pt:
                    return False

            snr = np.abs(pt["flux"].to_numpy()) / pt["fluxerr"].to_numpy()
            mask = snr >= sigma_limit

            if band is not None:
                if "band" not in pt:
                    return False
                mask = mask & (pt["band"].astype(str).to_numpy() == str(band))

            n = int(np.sum(mask))
            return (min_det <= n <= max_det)

        band_txt = f", band={band}" if band is not None else ""
        self.add_filter(
            _f,
            label=f"n_det ∈ [{min_det}, {max_det}] (σ≥{sigma_limit:g}{band_txt})",
        )

    def filter_time_since_last_det(
        self,
        min_days: float = 0.0,
        max_days: float = np.inf,
        sigma_limit: float = 5.0,
        band: str | None = None,
        now_jd: float | None = None,
    ) -> None:
        """
        Real time since last detection (days): now_jd - max(time_det).
        Detection defined as |flux|/fluxerr >= sigma_limit.
        """
        if now_jd is None:
            now_jd = Time.now().jd

        def _f(r: AmpelTransientReport) -> bool:
            if r.phot_table is None:
                r.create_table()
            pt = r.phot_table
            if pt is None or len(pt) == 0:
                return False

            for col in ("time", "flux", "fluxerr"):
                if col not in pt:
                    return False

            snr = np.abs(pt["flux"].to_numpy()) / pt["fluxerr"].to_numpy()
            mask = snr >= sigma_limit

            if band is not None:
                if "band" not in pt:
                    return False
                mask = mask & (pt["band"].astype(str).to_numpy() == str(band))

            if not np.any(mask):
                return False

            last = float(np.max(pt["time"].to_numpy()[mask]))
            dt_days = float(now_jd - last)
            return (min_days <= dt_days <= max_days)

        band_txt = f", band={band}" if band is not None else ""
        self.add_filter(
            _f,
            label=f"Δt(last det) ∈ [{min_days:g}, {max_days:g}] d (σ≥{sigma_limit:g}{band_txt})",
        )

    def filter_time_since_first_det(
        self,
        min_days: float = 0.0,
        max_days: float = np.inf,
        sigma_limit: float = 5.0,
        band: str | None = None,
        now_jd: float | None = None,
    ) -> None:
        """
        Real time since first detection (days): now_jd - min(time_det).
        Detection defined as |flux|/fluxerr >= sigma_limit.
        """
        if now_jd is None:
            now_jd = Time.now().jd

        def _f(r: AmpelTransientReport) -> bool:
            if r.phot_table is None:
                r.create_table()
            pt = r.phot_table
            if pt is None or len(pt) == 0:
                return False

            for col in ("time", "flux", "fluxerr"):
                if col not in pt:
                    return False

            snr = np.abs(pt["flux"].to_numpy()) / pt["fluxerr"].to_numpy()
            mask = snr >= sigma_limit

            if band is not None:
                if "band" not in pt:
                    return False
                mask = mask & (pt["band"].astype(str).to_numpy() == str(band))

            if not np.any(mask):
                return False

            first = float(np.min(pt["time"].to_numpy()[mask]))
            dt_days = float(now_jd - first)
            return (min_days <= dt_days <= max_days)

        band_txt = f", band={band}" if band is not None else ""
        self.add_filter(
            _f,
            label=f"Δt(first det) ∈ [{min_days:g}, {max_days:g}] d (σ≥{sigma_limit:g}{band_txt})",
        )

    def filter_host_separation_arcsec(
        self,
        min_arcsec: float = 0.0,
        max_arcsec: float = np.inf,
    ) -> None:
        """
        Filter on host[0]['distance'] in arcseconds (angular separation transient–host).
        """
        def _f(r: AmpelTransientReport) -> bool:
            if len(r.r.host) == 0:
                return False
            sep = r.r.host[0].distance  # arcsec
            if sep is None:
                return False

            try:
                sep = float(sep)
            except Exception:
                return False

            return (min_arcsec <= sep <= max_arcsec)

        self.add_filter(_f, label=f"host sep ∈ [{min_arcsec:g}, {max_arcsec:g}] arcsec")

    def filter_icecube_region(
        self,
        icecube_alert: Any,
        *,
        p_region: float = 0.9,
        map_dir: str = "./cache_ampelaccess/IceCubeAlerts",
        label: str | None = None,
        max_days_before: float | None = None,
        max_days_after: float | None = None,
    ) -> None: 
        """
        Filter on whether a transient is inside a given probability region of an IceCube alert's HEALPix skymap.
        Optionally, also filter on a time window around the alert's time, referred to first LSST alert of the transient. 
        """
        # resolve URL
        if hasattr(icecube_alert, "healpix_url"):
            map_url = str(getattr(icecube_alert, "healpix_url"))
        else:
            map_url = str(icecube_alert)

        # compute region pixels ONCE
        pix_sets = select_pixels_sets(
            map_url, p_region=p_region, map_dir=map_dir
        )

        # --- optional IC time ---
        use_time = (max_days_before is not None) or (max_days_after is not None)
        ic_jd = None

        if use_time:
            if max_days_before is None:
                max_days_before = 0.0
            if max_days_after is None:
                max_days_after = 0.0

            local_path = os.path.join(map_dir, map_url.split("/")[-1])
            try:
                hdr = fits.getheader(local_path)
                if "gps_time" in hdr and hdr["gps_time"] is not None:
                    ic_jd = float(
                        Time(float(hdr["gps_time"]), format="gps", scale="utc").jd
                    )
            except Exception:
                ic_jd = None

        def _f(r: "AmpelTransientReport") -> bool:

            # --- spatial check ---
            ra = float(r.r.object.ra)
            dec = float(r.r.object.dec)

            lon = ra * u.deg
            lat = dec * u.deg

            inside = False
            for ns, pixels in pix_sets.items():
                ip = int(
                    ah.lonlat_to_healpix(lon, lat, nside=int(ns), order="nested")
                )
                if ip in pixels:
                    inside = True
                    break

            if not inside:
                return False

            # --- optional time window check ---
            if use_time:

                if ic_jd is None:
                    return False  # strict

                if r.phot_table is None:
                    r.create_table()
                pt = r.phot_table

                if pt is None or len(pt) == 0 or "time" not in pt:
                    return False

                try:
                    first_jd = float(np.min(pt["time"]))
                except Exception:
                    return False

                dt = first_jd - ic_jd
                if not (-max_days_before <= dt <= max_days_after):
                    return False

            return True

        if label is None:
            short = map_url.split("/")[-1]
            label = f"IC {p_region:.2f} region ({short})"
            if use_time:
                label += f" & t0∈[-{max_days_before:g},+{max_days_after:g}]d"

        self.add_filter(_f, label=label)




    def apply_filter_preset(self, name: str, clear: bool = True, **overrides) -> None:
        """
        Apply a named configuration of existing filters for frequent science cases.

        name : str
            Preset name (case-insensitive). See _filter_presets() for the names!
        clear : bool
            If True, clears existing filters first.
        overrides :
            Override individual preset parameters (e.g. min_z=0.2, sigma_limit=7, ...) if desired.
        """
        key = name.strip().lower()
        presets = self._filter_presets()

        if key not in presets:
            raise ValueError(f"Unknown preset '{name}'. Available: {sorted(presets.keys())}")

        if clear:
            self.clear_filters()

        presets[key](self, **overrides)

    @staticmethod
    def _filter_presets():
        """
        Return dict mapping preset_name -> function(ars, **overrides) that calls existing filters.
        
        Note: The presets are preliminary!
        """
        def recent_young_highz_snia(
            ars: "AmpelReportSet",
            max_days_since_last: float = 3.0,
            max_days_since_first: float = 14.0,
            min_z: float = 0.1,
            min_p_snia: float = 0.35,
            band: str | None = None,
            now_jd: float | None = None,
        ) -> None:
            ars.filter_class_prob("SNIa", min_prob=min_p_snia)
            ars.filter_redshift(min_z=min_z)
            ars.filter_time_since_last_det(
                min_days=0.0,
                max_days=max_days_since_last,
                band=band,
                now_jd=now_jd,
            )
            ars.filter_time_since_first_det(
                min_days=0.0,
                max_days=max_days_since_first,
                band=band,
                now_jd=now_jd,
            )

        def decent_detections(
            ars: "AmpelReportSet",
            min_det: int = 3,
            max_det: int = 80,
            sigma_limit: float = 5.0,
            band: str | None = None,
        ) -> None:
            ars.filter_n_det(min_det=min_det, max_det=max_det, sigma_limit=sigma_limit, band=band)

        return {
            "recent_young_highz_snia": recent_young_highz_snia,
            "decent_detections": decent_detections,
        }

    def match_lsst_to_icecube_alerts(
        self,
        icecube_alerts: Any,
        *,
        p_region: float = 0.9,
        map_dir: str = "./cache_ampelaccess/IceCubeAlerts",
        use_all_reports: bool = False,
        return_obj_to_ics: bool = True,
        drop_empty: bool = True,
        verbose: bool = True,
        max_print: int = 30,
        max_days_before: float | None = None,
        max_days_after: float | None = None,
    ) -> tuple[dict[str, list[str]], dict[str, list[str]] | None]:
        """
        Match LSST reports to IceCube alerts given a probability region (p) of the HEALPix skymap.
        Optionally, also filter on a time window around the alert's time, referred to first LSST alert of the transient.
        Use all reports if use_all_reports=True, else only active ones.
        Return a dictionary mapping IC alert ID to a list of LSST object IDs.
        If return_obj_to_ics=True, return a second dictionary mapping LSST object ID to a list of IC alert IDs.
        Drop empty matches if drop_empty=True.
        Print some statistics if verbose=True.
        """
        # normalize alerts to list
        if not isinstance(icecube_alerts, (list, tuple)):
            icecube_alerts = [icecube_alerts]

        reports = self.reports if use_all_reports else self.active_reports

        use_time = (max_days_before is not None) or (max_days_after is not None)
        if use_time:
            if max_days_before is None:
                max_days_before = 0.0
            if max_days_after is None:
                max_days_after = 0.0

        ic_maps: list[tuple[str, dict[int, set[int]], float | None]] = []

        def _normalize_ic_id(x: Any, fallback: str) -> str:
            """
            Return a stable, hashable IceCube alert id.
            """
            if x is None:
                return str(fallback)

            # list/tuple sometimes occurs due to upstream parsing bugs
            if isinstance(x, (list, tuple)):
                for e in x:
                    if e is None:
                        continue
                    s = str(e).strip()
                    if s:
                        return s
                return str(fallback)

            s = str(x).strip()
            return s if s else str(fallback)

        for ic in icecube_alerts:
            if hasattr(ic, "healpix_url"):
                url = str(getattr(ic, "healpix_url"))
                fb = url.split("/")[-1]
                ic_id = _normalize_ic_id(getattr(ic, "event_name", None), fb)
                ic_alert_dt = getattr(ic, "alert_datetime", None)
            else:
                url = ic["healpix_url"] if isinstance(ic, dict) else str(ic)
                fb = url.split("/")[-1]
                ic_id = _normalize_ic_id(ic.get("event_name") if isinstance(ic, dict) else None, fb)
                ic_alert_dt = ic.get("alert_datetime") if isinstance(ic, dict) else None

            pix_sets = select_pixels_sets(url, p_region=p_region, map_dir=map_dir)

            ic_jd = None
            if use_time:
                # 1) ISO if present
                if ic_alert_dt:
                    try:
                        ic_jd = float(Time(str(ic_alert_dt), format="isot", scale="utc").jd)
                    except Exception:
                        ic_jd = None
                # 2) else FITS header gps_time (file should exist in map_dir)
                if ic_jd is None:
                    local_path = os.path.join(map_dir, url.split("/")[-1])
                    try:
                        hdr = fits.getheader(local_path)
                        if "gps_time" in hdr and hdr["gps_time"] is not None:
                            ic_jd = float(Time(float(hdr["gps_time"]), format="gps", scale="utc").jd)
                    except Exception:
                        ic_jd = None

            ic_maps.append((ic_id, pix_sets, ic_jd))


        def inside_pixsets(ra: float, dec: float, pix_sets: dict[int, set[int]]) -> bool:
            """
            Check if a given sky position (ra, dec) is inside a set of HEALPix pixels.
            """
            lon = ra * u.deg
            lat = dec * u.deg
            for ns, pixels in pix_sets.items():
                ip = int(ah.lonlat_to_healpix(lon, lat, nside=int(ns), order="nested"))
                if ip in pixels:
                    return True
            return False

        ic_to_objs: dict[str, list[str]] = {ic_id: [] for ic_id, _, _ in ic_maps}
        obj_to_ics: dict[str, list[str]] = {} if return_obj_to_ics else None

        for r in reports:
            obj_id = str(r.r.object.id)
            ra = float(r.r.object.ra)
            dec = float(r.r.object.dec)

            first_jd = None
            if use_time:
                if r.phot_table is None:
                    r.create_table()
                pt = r.phot_table
                if pt is None or len(pt) == 0 or "time" not in pt:
                    continue
                try:
                    first_jd = float(np.min(pt["time"]))
                except Exception:
                    continue

            hits: list[str] = []
            for ic_id, pix_sets, ic_jd in ic_maps:
                if not inside_pixsets(ra, dec, pix_sets):
                    continue

                if use_time:
                    if ic_jd is None:
                        continue  # strict: skip this IC alert
                    dt = float(first_jd) - float(ic_jd)
                    if not (-max_days_before <= dt <= max_days_after):
                        continue

                hits.append(ic_id)

            if hits:
                for ic_id in hits:
                    ic_to_objs[ic_id].append(obj_id)
                if obj_to_ics is not None:
                    obj_to_ics[obj_id] = hits

        if drop_empty:
            ic_to_objs = {k: v for k, v in ic_to_objs.items() if v}

        if verbose:
            scope = "all reports" if use_all_reports else "active_reports"
            nrep = len(reports)
            print(f"IC matching p={p_region:g} on {scope}: {nrep} LSST reports, {len(ic_maps)} IC alerts")
            items = sorted(ic_to_objs.items(), key=lambda kv: len(kv[1]), reverse=True)
            for i, (ic_id, objs) in enumerate(items[:max_print]):
                print(f"  {ic_id}: {len(objs)} matches")
            if len(items) > max_print:
                print(f"  ... ({len(items) - max_print} more IC alerts with matches)")

        return ic_to_objs, obj_to_ics
    
    def plot_sky_mollweide(
        self,
        *,
        icecube_alerts: Any | None = None,
        use_all_reports: bool = False,
        figsize: tuple[float, float] = (7.0, 4.5),
        s_lsst: float = 8.0,
        s_ic: float = 45.0,
        alpha_lsst: float = 0.9,
        alpha_ic: float = 1.0,
        map_dir: str = "./cache_ampelaccess/IceCubeAlerts",
        save_path: str | None = None,
        show: bool = True,
    ):
        """
        Plot LSST alert positions in a Mollweide all-sky projection.
        Optionally overlay IceCube alert centers (from alert dict / IceCubeAlert / healpix_url).

        Parameters
        ----------
        icecube_alerts:
            None, a single alert, or iterable of alerts.
            Supported alert forms:
            - IceCubeAlert (has .healpix_url and optionally .raw)
            - dict with keys containing ra/dec or healpix_url
            - str healpix_url
        use_all_reports:
            If True, plot all reports; otherwise only active_reports.
        map_dir:
            Cache directory for downloading IC healpix maps when ra/dec not given.
        save_path:
            If set, save figure to this path.
        """

        # Collect LSST positions
        reports = self.reports if use_all_reports else self.active_reports
        if not reports:
            raise RuntimeError("No reports to plot (empty active_reports/reports).")

        ra_lsst = np.array([float(r.r.object.ra) for r in reports], dtype=float)
        dec_lsst = np.array([float(r.r.object.dec) for r in reports], dtype=float)

        
        # Helper: extract IC center
        def _is_url(s: str) -> bool:
            return str(s).startswith("http://") or str(s).startswith("https://")

        def _download_to(url: str, out_dir: str) -> str:
            """
            Download URL to out_dir or return local path if already a file.

            Notes
            -----
            - If `url` is a local existing file path, return it unchanged.
            - If `url` is an http(s) URL, download it to out_dir and return the local path.
            """
            # local file path?
            p = Path(str(url))
            if p.exists() and p.is_file():
                return str(p)

            if not _is_url(str(url)):
                raise ValueError(f"Not a URL and not an existing file: {url!r}")

            os.makedirs(out_dir, exist_ok=True)
            fn = str(url).split("/")[-1]
            path = os.path.join(out_dir, fn)

            r = requests.get(str(url), timeout=30)
            r.raise_for_status()
            with open(path, "wb") as f:
                f.write(r.content)
            return path

        def _ic_center(alert: Any) -> tuple[float, float] | None:
            """
            Extract the center coordinates (ra, dec) from an alert.
            """
            # --- 1) raw dict fields (if available)
            if hasattr(alert, "raw"):
                raw = getattr(alert, "raw")
                if isinstance(raw, dict):
                    for k_ra, k_dec in [("ra", "dec"), ("RA", "DEC"), ("right_ascension", "declination")]:
                        if k_ra in raw and k_dec in raw:
                            try:
                                return float(raw[k_ra]), float(raw[k_dec])
                            except Exception:
                                pass

            if isinstance(alert, dict):
                for k_ra, k_dec in [("ra", "dec"), ("RA", "DEC"), ("right_ascension", "declination")]:
                    if k_ra in alert and k_dec in alert:
                        try:
                            return float(alert[k_ra]), float(alert[k_dec])
                        except Exception:
                            pass

            # --- 2) healpix_url or path
            if hasattr(alert, "healpix_url"):
                url_or_path = str(getattr(alert, "healpix_url"))
            elif isinstance(alert, dict) and "healpix_url" in alert:
                url_or_path = str(alert["healpix_url"])
            elif isinstance(alert, str):
                url_or_path = str(alert)
            else:
                return None

            try:
                map_path = _download_to(url_or_path, map_dir)
            except Exception:
                return None

            # --- 3) Coordinates from header
            try:
                hdr0 = fits.getheader(map_path)
                for k_ra, k_dec in [("RA", "DEC"), ("RA_OBJ", "DEC_OBJ"), ("OBJRA", "OBJDEC")]:
                    if k_ra in hdr0 and k_dec in hdr0:
                        return float(hdr0[k_ra]), float(hdr0[k_dec])
            except Exception:
                pass

            # --- 4) robust fallback: Highest probability pixel
            try:
                skymap = QTable.read(map_path)
                if "PROBDENSITY" not in skymap.colnames or "UNIQ" not in skymap.colnames:
                    return None

                i_max = int(np.argmax(skymap["PROBDENSITY"]))
                level, ipix = ah.uniq_to_level_ipix(skymap["UNIQ"][i_max])
                nside = ah.level_to_nside(level)

                lon, lat = ah.healpix_to_lonlat(ipix, nside=int(nside), order="nested")
                return float(lon.to_value(u.deg)), float(lat.to_value(u.deg))

            except Exception:
                return None



        # Normalize icecube_alerts to list
        ic_points: list[tuple[float, float]] = []
        if icecube_alerts is not None:
            if not isinstance(icecube_alerts, (list, tuple)):
                icecube_alerts = [icecube_alerts]
            for ic in icecube_alerts:
                c = _ic_center(ic)
                if c is not None:
                    ic_points.append(c)

        # CORE: Plot Mollweide
        fig = plt.figure(figsize=figsize, dpi=100)
        ax = plt.axes(
            [0.06, 0.08, 0.88, 0.86],
            projection="astro degrees mollweide",
            center=SkyCoord(0 * u.deg, 0 * u.deg, frame="icrs"),
        )

        ax.grid()
        ax.locator_params(nbins=10)

        # LSST points
        ax.scatter(
            ra_lsst,
            dec_lsst,
            s=s_lsst,
            alpha=alpha_lsst,
            transform=ax.get_transform("world"),
            label=f"LSST ({len(ra_lsst)})",
        )

        # IC centers (optional)
        if ic_points:
            ra_ic = [p[0] for p in ic_points]
            dec_ic = [p[1] for p in ic_points]
            ax.scatter(
                ra_ic,
                dec_ic,
                s=s_ic,
                marker="x",
                linewidths=2.0,
                alpha=alpha_ic,
                transform=ax.get_transform("world"),
                label=f"IceCube centers ({len(ic_points)})",
            )

        ax.legend(loc="upper right", framealpha=0.95, fontsize=9)

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight")

        if show:
            plt.show()

        return fig, ax





    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def reports(self) -> List[AmpelTransientReport]:
        return list(self._reports.values())

    @property
    def active_reports(self) -> List[AmpelTransientReport]:
        return [self._reports[i] for i in self._active_ids]

    # ------------------------------------------------------------------
    # Minimal summary visualization
    # ------------------------------------------------------------------

    def plot_lightcurves(
        self,
        *,
        grid: bool = False,
        ncols: int = 3,
        figsize: tuple | None = (15, 10),
        row_height: float = 3.2,
        col_width: float = 5.0,
        constrained: bool = True,
        **kwargs,
    ):
        """
        Plot lightcurves of active reports.
        Use AmpelReport.plot_lightcurve() for every report.

        grid=False: one plot per report (list of axes)
        grid=True:  single figure with grid layout (fig, axes)

        Notes
        -----
        For large N, a fixed `figsize` will squeeze rows and can lead to overlapping.
        If `figsize` is None, the height is scaled as (row_height * nrows) and
        the width as (col_width * ncols).
        """
        reports = self.active_reports
        if not reports:
            raise RuntimeError("No active reports to plot")

        if not grid:
            axes = []
            for r in reports:
                axes.append(r.plot_lightcurve(**kwargs))
            return axes

        n = len(reports)
        nrows = math.ceil(n / ncols)

        # Auto-figsize for large grids
        if figsize is None:
            figsize = (col_width * ncols, row_height * nrows)

        fig = plt.figure(figsize=figsize, constrained_layout=bool(constrained))
        gs = GridSpec(nrows, ncols, figure=fig)

        axes = []
        for i, r in enumerate(reports):
            row, col = divmod(i, ncols)
            ax = fig.add_subplot(gs[row, col])
            r.plot_lightcurve(ax=ax, **kwargs)
            axes.append(ax)

        # turn off unused cells
        for j in range(i + 1, nrows * ncols):
            row, col = divmod(j, ncols)
            fig.add_subplot(gs[row, col]).axis("off")

        # Avoid tight_layout with large grids; constrained_layout is more robust
        if not constrained:
            fig.tight_layout()

        return fig, axes



    def show_classifications(
        self,
        *,
        grid: bool = False,
        ncols: int = 3,
        figsize: tuple | None = (15, 10),
        row_height: float = 3.6,
        col_width: float = 4.5,
        threshold: float = 0.01,
        constrained: bool = True,
    ):
        """
        Show classification diagrams of active reports.
        Use AmpelReport.show_classification() for every report.

        grid=False: one radar per report
        grid=True:  radar plots in a grid

        """
        reports = self.active_reports
        if not reports:
            raise RuntimeError("No active reports to show")

        if not grid:
            axes = []
            for r in reports:
                axes.append(r.show_classification(threshold=threshold))
            return axes

        n = len(reports)
        nrows = math.ceil(n / ncols)

        if figsize is None:
            figsize = (col_width * ncols, row_height * nrows)

        fig = plt.figure(figsize=figsize, constrained_layout=bool(constrained))
        gs = GridSpec(nrows, ncols, figure=fig)

        axes = []
        for i, r in enumerate(reports):
            row, col = divmod(i, ncols)
            ax = fig.add_subplot(gs[row, col], projection="polar")
            r.show_classification(ax=ax,threshold=threshold)
            try:
                name = r._display_name()
            except Exception:
                name = str(r.r.object.id)
            ax.set_title(str(name), fontsize=9, pad=6)
            axes.append(ax)

        for j in range(i + 1, nrows * ncols):
            row, col = divmod(j, ncols)
            fig.add_subplot(gs[row, col]).axis("off")

        if not constrained:
            fig.tight_layout()

        return fig, axes



    def show_hosts(
        self,
        *,
        grid: bool = False,
        ncols: int = 2,
        figsize: tuple | None = (16, 6),
        row_height: float = 3.4,
        col_width: float = 8.0,
        constrained: bool = True,
        hspace: float = 0.25,
        wspace: float = 0.15,
    ):
        """
        Show host / object information of active reports.
        Use AmpelReport.show_hostinfo() for every report.

        grid=False: one figure per report
        grid=True:  host info arranged in a grid (subfigures)

        Notes
        -----
        For large N, a fixed `figsize` will squeeze rows and can lead to overlapping.
        If `figsize` is None, the height is scaled as (row_height * nrows) and
        the width as (col_width * ncols).
        """
        reports = self.active_reports
        if not reports:
            raise RuntimeError("No active reports to show")

        if not grid:
            figs = []
            for r in reports:
                figs.append(r.show_hostinfo())
            return figs

        n = len(reports)
        nrows = math.ceil(n / ncols)

        if figsize is None:
            figsize = (col_width * ncols, row_height * nrows)

        fig = plt.figure(figsize=figsize, constrained_layout=bool(constrained))
        outer = GridSpec(
            nrows,
            ncols,
            figure=fig,
            hspace=hspace,
            wspace=wspace,
        )

        for i, r in enumerate(reports):
            row, col = divmod(i, ncols)
            subfig = fig.add_subfigure(outer[row, col])
            r.show_hostinfo(fig=subfig)

        if not constrained:
            fig.tight_layout()

        return fig





    # ------------------------------------------------------------------
    # Row-wise summary plotting
    # ------------------------------------------------------------------

    def plot_summary_rows(
        self,
        figsize: tuple = (18, 4),
        lc_kwargs: Optional[dict] = None,
        max_rows: Optional[int] = None,
        cls_threshold: float = 0.01,
        *,
        classifier: int | str = 0,
        model: int = 0,
    ):
        """
        Plot the complete summary of all active reports including lightcurve, classification, finder and additional info.
        Uses AmpelReport.draw_summary_row() for every report.
        """

        lc_kwargs = lc_kwargs or {}

        reports = self.active_reports
        if max_rows is not None:
            reports = reports[:max_rows]

        nrows = len(reports)
        if nrows == 0:
            raise RuntimeError("No active reports to plot")

        fig = plt.figure(figsize=(figsize[0], figsize[1] * nrows), constrained_layout=False)
        outer = GridSpec(nrows=nrows, ncols=1, figure=fig, hspace=0.35)

        axes: dict[str, dict[str, Any]] = {}
        for i, report in enumerate(reports):
            obj_id = str(report.r.object.id)
            axes[obj_id] = {}
            subfig = fig.add_subfigure(outer[i, 0])

            report.draw_summary_row(
                subfig,
                lc_kwargs=lc_kwargs,
                cls_threshold=cls_threshold,
                classifier=classifier,
                model=model,
            )
            axes[obj_id]["subfig"] = subfig

        return fig, axes