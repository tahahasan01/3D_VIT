"""Garment data models.

Defines the schema for garment types and measurements used
when converting a 2D outfit image into a 3D garment mesh.
"""

from enum import Enum

from pydantic import BaseModel, Field


class GarmentType(str, Enum):
    """Supported garment template types."""

    TSHIRT = "tshirt"
    POLO = "polo"
    BUTTON_DOWN = "button_down"
    HOODIE = "hoodie"
    JACKET = "jacket"
    PANTS = "pants"
    DRESS = "dress"


class GarmentMeasurements(BaseModel):
    """Measurements for sizing a 3D garment.

    Not all fields are required for every garment type.
    Provide the fields relevant to the selected garment_type.
    """

    garment_type: GarmentType = Field(
        description="Type of garment template to use",
    )

    # -- Shared measurements --------------------------------------------------
    length_cm: float | None = Field(
        default=None,
        ge=20.0,
        le=150.0,
        description="Garment length in centimeters",
    )

    # -- Top / dress measurements ---------------------------------------------
    chest_cm: float | None = Field(
        default=None,
        ge=60.0,
        le=160.0,
        description="Chest circumference (tops and dresses)",
    )
    sleeve_length_cm: float | None = Field(
        default=None,
        ge=10.0,
        le=90.0,
        description="Sleeve length (tops only)",
    )

    # -- Bottom / dress measurements ------------------------------------------
    waist_cm: float | None = Field(
        default=None,
        ge=50.0,
        le=150.0,
        description="Waist circumference (pants and dresses)",
    )
    hip_cm: float | None = Field(
        default=None,
        ge=60.0,
        le=160.0,
        description="Hip circumference (pants and dresses)",
    )
    inseam_cm: float | None = Field(
        default=None,
        ge=20.0,
        le=100.0,
        description="Inseam length (pants only)",
    )

    # -- Body landmarks (optional, from SMPL body generation) --------------------
    body_landmarks: dict | None = Field(
        default=None,
        description="Body landmark positions from SMPL body generation (auto-populated by frontend)",
    )
    # -- Body measurements (optional, for conforming garment from body mesh) ------
    body_measurements: dict | None = Field(
        default=None,
        description="Body measurements used to generate the body; when provided with body_landmarks, garment is built as conforming offset from body mesh",
    )

    model_config = {"json_schema_extra": {"examples": [
        {
            "garment_type": "tshirt",
            "chest_cm": 100,
            "length_cm": 72,
            "sleeve_length_cm": 24,
        }
    ]}}
