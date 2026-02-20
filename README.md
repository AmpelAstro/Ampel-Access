# Ampel-Access
Methods for accessing, parsing and using Ampel outputs.


## AMPEL Report Visualization Tools

Tools for inspecting, filtering, and visualizing AMPEL LSST alert reports, including lightcurves, classification probabilities, and host/environment context.

This repository provides three main classes:

* **`AmpelTransientReport`** – a wrapper around a single AMPEL `LSSTReport`
* **`AmpelReportSet`** – a container for streaming, caching, managing, filtering, and batch-plotting multiple reports
* **`IceCubeAlert`** – a wrapper for loading, displaying and crossmatching IceCube alerts

The code is intended for **interactive analysis**, vetting pipelines, and human-in-the-loop review of transient alerts that were provided as **output** of full AMPEL analysis jobs.

The notebook **`demo_AmpelAccess`** demonstrates how AMPEL Reports published through Hopskotch can be loaded, cached, updated incrementally, filtered, and visualized. It also shows how to incorporate **IceCube alerts** for basic multi-messenger matching



---

## Core Concepts

### `AmpelTransientReport`

A lightweight wrapper around a single AMPEL `LSSTReport` providing:

* Parsed access to photometry, classifications, and host info
* Convenience plotting methods
* Finder chart loading and normalization
* Compact inspection helper for notebooks 
* Displaying Classifier/model

```python
from AmpelReport import AmpelTransientReport

report = AmpelTransientReport(lsst_report)
report.plot_lightcurve()
report.show_classification()
report.show_hostinfo()
report.plot_summary_row()
```

---

### `AmpelReportSet`

A container for **many** `AmpelTransientReport` objects with:

-   Streaming updates from Hopskotch with incremental offsets via Kafka consumer groups
-   Local caching of per-object merged reports (pickles) plus metadata (`meta.json`)
-   Report merging logic (photometry deduplication by time/band, "latest-wins" for other blocks)
-   Dynamic filtering of the active subset based on science-driven criteria
-   Batch plotting utilities
-   Row-wise "review table" summary figures for human vetting
-   Optional IceCube sky-map matching + filtering (multi-order HEALPix maps)

```python
from AmpelReportSet import AmpelReportSet

ars = AmpelReportSet(reports)
ars.filter_class_prob("SN Ia", min_prob=0.5, classifier=0, model=0)
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
    classifier=0,
    model=0
)
```

--- 

### `IceCubeAlert`

A convenience wrapper for a single IceCube alert (public ROC skymap or Kafka message). It encapsulates:

-   Either downloading of public alerts or parsing of alert dictionaries or local cached `.fits.gz`
-   Plotting of multi-order HEALPix probability maps  
-   Probability-region containment checks for given sky coordinates or LSST reports 
-   Optional visualization of LSST candidates on the skymap  

```python
from icecube_tools import IceCubeAlert

ic_alert = IceCubeAlert.from_dict(alert_dict)
ic_alert.plot(r=3.0, show_alert_datetime=True)

inside = ic_alert.contains_source(
    ra=243.7,
    dec=6.8,
    p_region=0.9,
)

ic_to_objs, obj_to_ics = ars.match_lsst_to_icecube_alerts(
    ic_alert,
    p_region=0.9,
    max_days_before=10,
    max_days_after=30,
)
```


## Installation

Ampel-Access is a light-weight build that runs without requiring a full AMPEL installation.  
`LSSTReportModel` handles incoming AMPEL reports.

### Step-by-step setup
1. **Download the repository**

    ```bash
    cd Your/Desired/Directory 
    git clone https://github.com/AmpelAstro/Ampel-Access.git
    ```

2. **Create a fresh Conda environment**

    ```bash
    conda create -y -n ampelaccess_env python=3.10 
    conda activate ampelaccess_env
    ```

3. **Install core dependencies**

    ```bash
    conda install -y -c conda-forge \
        "numpy<2" \
        pandas \
        matplotlib \
        astropy \
        healpy \
        ligo.skymap \
        requests \
        "pydantic>=2"
    ```

4. **Install additional Python packages** 

    ```bash
    python -m pip install "mhealpy==0.3.6" 
    python -m pip install hop-client
    ```

5. **Install Jupyter kernel support**

    ```bash
    python -m pip install ipykernel
    python -m ipykernel install --user --name ampelaccess_env --display-name "ampelaccess_env"
    ```

6. **Open the demo notebook**

    Open `demo_AmpelAccess.ipynb` 

    Select the conda kernel: `ampelaccess_env`
    