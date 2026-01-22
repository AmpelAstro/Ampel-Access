# Ampel-Access
Methods for accessing, parsing and using Ampel outputs.


## AMPEL Report Visualization Tools

Tools for inspecting, filtering, and visualizing AMPEL LSST alert reports, including lightcurves, classification probabilities, and host/environment context.

This repository provides two main classes:

* **`AmpelTransientReport`** – a wrapper around a single AMPEL `LSSTReport`
* **`AmpelReportSet`** – a container for managing, filtering, and batch-plotting multiple reports

The code is intended for interactive analysis**, vetting pipelines, and human-in-the-loop review of transient alerts that were provided as **output** of full AMPEL analysis jobs.

The notebook **`demo_AmpelAccess`** demonstrates how AMPEL Reports published through Hopskotch can be laoded as AmpelTransientReports.



---

## Core Concepts

### `AmpelTransientReport`

A lightweight wrapper around a single AMPEL `LSSTReport` providing:

* Parsed access to photometry, classifications, and host info
* Convenience plotting methods
* Finder chart loading and normalization

```python
from AmpelReport import AmpelTransientReport

report = AmpelTransientReport(lsst_report)
report.plot_lightcurve()
report.show_classification()
report.show_hostinfo()
```

---

### `AmpelReportSet`

A container for **many** `AmpelTransientReport` objects with filtering and batch visualization.

```python
from AmpelReportSet import AmpelReportSet

ars = AmpelReportSet(reports)
ars.filter_class_prob("SN Ia", min_prob=0.5)
ars.filter_age(
    min_tago=0,
    max_tago=30,
    min_duration=1,
    sigma_limit=3.0,
)
ars.print_status()
fig, axes = ars.plot_summary_rows(
    max_rows=10,
    lc_kwargs={"sigma_limit": 3},
)
```




# Todo
- Add method to read Hopskotch topic directly from AmpelReportSet
- Store / archive loaded data and correctly merge with new reports.
- Nomenclature? Report, Set etc?
- Load LSST thumbnails (from somewhere...).
- Fix catalog cutout bug.
- Graceful checks for format
- Correctly display LSST name.
- Proper poetry install + access Ampel-hu-astro methods without reading the full thing.
- Add healpix sky selection, possibly by pointing to online MM alert file.
- Store a set of filter as a configuration?
- Filter based on association with specific MM alert (based on properties)
- Deal with multiple classifications in a reasonable way.
- Is there a way to incorporate different states? Or just grab the last consistently?

