# LSSTReportModel.py
from __future__ import annotations
from typing import Sequence
from pydantic import BaseModel, ConfigDict, Field

"""
LSSTReportModel
===============

Lightweight Pydantic models defining a representation of LSST alert report content.
It mirrors the output of Ampel's LSSTReport. 

This module provides data containers for:
- Object metadata
- Photometric time-series
- Classifier outputs and probabilities
- Host galaxy information
- Derived feature tables

The `LSSTReport` model serves as a data structure for downstream analysis, filtering, and visualization.
"""


class _Base(BaseModel):
    # "ignore" makes this robust if upstream adds keys you don't care about
    model_config = ConfigDict(
        extra="ignore",
        arbitrary_types_allowed=True,
        validate_default=True,
        populate_by_name=True,
    )

class PhotometricPoint(_Base):
    time: float = Field(description="epoch of observation (JD)")
    flux: float = Field(description="observed flux")
    fluxerr: float = Field(description="flux uncertainty")
    band: str = Field(description="photometric band")
    zp: float = Field(description="zero point")
    zpsys: str = Field(description="zero point system")

class Object(_Base):
    id: int = Field(description="diaObjectId")
    external_id: str | None = Field(default=None, description="External Object ID")
    ra: float = Field(description="right ascension (deg)")
    dec: float = Field(description="declination (deg)")
    source: str = Field(description="data source")

class ModelClassification(_Base):
    model: str
    probabilities: dict[str, float]

class Classification(_Base):
    name: str
    version: str
    info: str | None = None
    models: list[ModelClassification]

class Host(_Base):
    name: str
    source: str
    redshift: float
    redshift_error: float | None = None
    distance: float
    info: str | None = None

class Feature(_Base):
    name: str
    version: str
    info: str | None = None
    features: dict[str, float]

class LSSTReport(_Base):
    object: Object
    state: int
    photometry: Sequence[PhotometricPoint]
    classification: list[Classification] = []
    host: list[Host] = []
    features: list[Feature] = []
