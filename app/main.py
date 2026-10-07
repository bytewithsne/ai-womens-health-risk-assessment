from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import json
from pathlib import Path


app = FastAPI(
    title="Women's Health PSPI API",
    description=(
        "Research prototype for personalized symptom-pattern "
        "assessment using the Personalized Symptom-Pattern Index (PSPI)."
    ),
    version="3.0.0"
)


REFERENCE_FILE = Path(
    "data/processed/pspi_reference_parameters.json"
)


with open(REFERENCE_FILE, "r", encoding="utf-8") as f:
    REFERENCE = json.load(f)


class PSPIInput(BaseModel):
    bleeding_mean: Optional[float] = None
    general_emotional_condition_mean: Optional[float] = None
    general_physical_condition_mean: Optional[float] = None
    type_of_stool_mean: Optional[float] = None
    lower_back_pain_mean: Optional[float] = None
    headache_mean: Optional[float] = None

    bleeding_historical_deviation: Optional[float] = None
    general_emotional_condition_historical_deviation: Optional[float] = None
    general_physical_condition_historical_deviation: Optional[float] = None
    type_of_stool_historical_deviation: Optional[float] = None
    ovulation_test_values_historical_deviation: Optional[float] = None


CURRENT_FEATURES = [
    "bleeding_mean",
    "general_emotional_condition_mean",
    "general_physical_condition_mean",
    "type_of_stool_mean",
    "lower_back_pain_mean",
    "headache_mean",
]

DEVIATION_FEATURES = [
    "bleeding_historical_deviation",
    "general_emotional_condition_historical_deviation",
    "general_physical_condition_historical_deviation",
    "type_of_stool_historical_deviation",
    "ovulation_test_values_historical_deviation",
]


HIGHER_BURDEN_FEATURES = [
    "bleeding_mean",
    "type_of_stool_mean",
    "lower_back_pain_mean",
    "headache_mean",
]


LOWER_BURDEN_FEATURES = [
    "general_emotional_condition_mean",
    "general_physical_condition_mean",
]


def minmax(value, minimum, maximum):

    if maximum == minimum:
        return 0.0

    normalized = (value - minimum) / (maximum - minimum)

    return max(0.0, min(1.0, normalized))


def calculate_current_burden(data):

    values = []

    for feature in CURRENT_FEATURES:

        value = getattr(data, feature)

        if value is None:
            continue

        reference = REFERENCE["current_features"][feature]

        normalized = minmax(
            value,
            reference["min"],
            reference["max"]
        )

        if feature in LOWER_BURDEN_FEATURES:
            normalized = 1.0 - normalized

        values.append(normalized)

    if not values:
        return None

    return sum(values) / len(values)


def calculate_historical_deviation(data):

    values = []

    for feature in DEVIATION_FEATURES:

        value = getattr(data, feature)

        if value is None:
            continue

        reference = REFERENCE[
            "historical_deviation_features"
        ][feature]

        normalized = minmax(
            abs(value),
            0.0,
            max(
                abs(reference["min"]),
                abs(reference["max"])
            )
        )

        values.append(normalized)

    if not values:
        return None

    return sum(values) / len(values)


@app.get("/")
def home():

    return {
        "message": "Women's Health PSPI API is running!",
        "status": "success",
        "purpose": (
            "Research prototype for personalized "
            "symptom-pattern assessment"
        ),
        "index": "Personalized Symptom-Pattern Index (PSPI)",
        "warning": (
            "This prototype is not a medical diagnosis "
            "and is not clinically validated."
        )
    }


@app.post("/assess-pspi")
def assess_pspi(data: PSPIInput):

    current_burden = calculate_current_burden(data)

    historical_deviation = calculate_historical_deviation(data)

    components = []

    if current_burden is not None:
        components.append(current_burden)

    if historical_deviation is not None:
        components.append(historical_deviation)

    if not components:

        return {
            "personalized_symptom_pattern_index": None,
            "pattern_category": "Insufficient evidence",
            "evidence_components_available": 0,
            "warning": (
                "Insufficient evidence to calculate PSPI."
            )
        }

    pspi = sum(components) / len(components) * 100

    if pspi < 33:
        category = "Lower pattern"
    elif pspi < 66:
        category = "Intermediate pattern"
    else:
        category = "Higher pattern"

    evidence_count = (
        sum(
            getattr(data, feature) is not None
            for feature in CURRENT_FEATURES
        )
        +
        sum(
            getattr(data, feature) is not None
            for feature in DEVIATION_FEATURES
        )
    )

    evidence_coverage = (
        evidence_count / 11
    ) * 100

    return {
        "personalized_symptom_pattern_index": round(
            pspi, 2
        ),
        "pattern_category": category,
        "current_burden_component": (
            round(current_burden * 100, 2)
            if current_burden is not None
            else None
        ),
        "historical_deviation_component": (
            round(historical_deviation * 100, 2)
            if historical_deviation is not None
            else None
        ),
        "evidence_coverage_percent": round(
            evidence_coverage, 2
        ),
        "evidence_components_available": evidence_count,
        "warning": (
            "This is a research-derived symptom-pattern "
            "representation, not a clinical diagnosis."
        )
    }