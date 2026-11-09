from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field, field_validator

SummaryType = Literal[
    "Breezy", "Breezy and Dry", "Breezy and Foggy", "Breezy and Mostly Cloudy",
    "Breezy and Overcast", "Breezy and Partly Cloudy", "Clear",
    "Dangerously Windy and Partly Cloudy", "Drizzle", "Dry", "Dry and Mostly Cloudy",
    "Dry and Partly Cloudy", "Foggy", "Humid and Mostly Cloudy", "Humid and Overcast",
    "Humid and Partly Cloudy", "Light Rain", "Mostly Cloudy", "Overcast", "Partly Cloudy",
    "Rain", "Windy", "Windy and Dry", "Windy and Foggy", "Windy and Mostly Cloudy",
    "Windy and Overcast", "Windy and Partly Cloudy",
]


class WeatherInput(BaseModel):
    """One hourly weather observation (same fields as weatherHistory.csv)."""

    formatted_date: str = Field(
        description="Observation timestamp with UTC offset, e.g. '2016-09-09 14:00:00.000 +0200'",
        examples=["2016-09-09 14:00:00.000 +0200"],
    )
    summary: SummaryType = Field(description="Hourly summary, e.g. 'Partly Cloudy'")
    precip_type: Literal["rain", "snow"] | None = Field(
        default=None, description="'rain', 'snow' or null when unknown"
    )
    temperature_c: float = Field(ge=-50, le=60)
    apparent_temperature_c: float = Field(ge=-60, le=60)
    humidity: float = Field(ge=0.0, le=1.0, description="Relative humidity as a 0-1 fraction")
    wind_speed_kmh: float = Field(ge=0, le=150)
    wind_bearing_degrees: float = Field(ge=0, le=360)
    visibility_km: float = Field(ge=0)
    pressure_millibars: float = Field(ge=0)

    @field_validator("formatted_date")
    @classmethod
    def validate_timestamp(cls, value: str) -> str:
        try:
            pd.to_datetime(value, utc=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("formatted_date must be a valid timestamp with a UTC offset") from exc
        return value


class ClassProbability(BaseModel):
    daily_summary: str
    probability: float


class PredictionResponse(BaseModel):
    prediction: int
    daily_summary: str
    probability: float
    top_3: list[ClassProbability]


class ErrorDetail(BaseModel):
    field: str
    message: str


class ErrorResponse(BaseModel):
    status: str
    message: str
    details: list[ErrorDetail]


class BatchPredictionRequest(BaseModel):
    observations: list[WeatherInput] = Field(min_length=1)


class BatchPredictionResponse(BaseModel):
    count: int
    predictions: list[PredictionResponse]
