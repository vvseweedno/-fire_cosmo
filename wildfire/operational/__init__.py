"""Operational ingestion/event interfaces kept separate from competition scoring."""

from wildfire.operational.ingest import (
    BurnObservationPair,
    ObservationDescriptor,
    build_burn_pair,
    operational_capabilities,
    validate_observation,
)

__all__ = [
    "BurnObservationPair",
    "ObservationDescriptor",
    "build_burn_pair",
    "operational_capabilities",
    "validate_observation",
]
