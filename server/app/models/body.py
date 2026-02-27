"""Body measurement data models.

Defines the schema for body measurements used to generate
a parametric 3D human model.
"""

from enum import Enum

from pydantic import BaseModel, Field


class Gender(str, Enum):
    """Gender selection for base body proportions."""

    MALE = "male"
    FEMALE = "female"


class BodyMeasurements(BaseModel):
    """Input measurements for 3D body generation.

    All length values are in centimeters. Circumference values
    represent the full wrap-around measurement.
    """

    gender: Gender = Field(
        description="Gender for base body proportions",
    )
    height_cm: float = Field(
        ge=140.0,
        le=220.0,
        description="Total height in centimeters",
    )
    chest_cm: float = Field(
        ge=60.0,
        le=160.0,
        description="Chest circumference in centimeters",
    )
    waist_cm: float = Field(
        ge=50.0,
        le=150.0,
        description="Waist circumference in centimeters",
    )
    hip_cm: float = Field(
        ge=60.0,
        le=160.0,
        description="Hip circumference in centimeters",
    )
    shoulder_width_cm: float = Field(
        ge=30.0,
        le=60.0,
        description="Shoulder-to-shoulder distance in centimeters",
    )
    arm_length_cm: float = Field(
        ge=40.0,
        le=90.0,
        description="Shoulder to wrist length in centimeters",
    )
    inseam_cm: float = Field(
        ge=60.0,
        le=100.0,
        description="Crotch to ankle length in centimeters",
    )
    use_base_mesh: bool = Field(
        default=False,
        description="If true, use the built-in male base mesh (OBJ) scaled to height instead of parametric generation",
    )
    use_smpl: bool = Field(
        default=False,
        description="If true, use SMPL body model (male/female) from assets/smpl, scaled to height; requires SMPL registration",
    )
    skin_color_hex: str | None = Field(
        default=None,
        description="Optional skin color as hex e.g. '#DEC3AA'. When set, used as body base color in GLB.",
    )

    model_config = {"json_schema_extra": {"examples": [
        {
            "gender": "male",
            "height_cm": 175,
            "chest_cm": 96,
            "waist_cm": 82,
            "hip_cm": 98,
            "shoulder_width_cm": 45,
            "arm_length_cm": 60,
            "inseam_cm": 80,
        }
    ]}}
