"""
Feature Mapping and Validation Service for AutoRisk AI.

Translates between user/document input keys and Part 1's 26 canonical model features,
strips identifiers, enforces domain constraints, and guards against invalid mappings.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from backend.app.schemas import UNSUPPORTED_FRONTEND_FIELDS


# Canonical 26 features expected by the trained pipeline
CANONICAL_FEATURES = [
    "WeekOfMonth",
    "Age",
    "Deductible",
    "DriverRating",
    "Month",
    "DayOfWeek",
    "Make",
    "AccidentArea",
    "Sex",
    "MaritalStatus",
    "Fault",
    "PolicyType",
    "VehicleCategory",
    "VehiclePrice",
    "Days_Policy_Accident",
    "Days_Policy_Claim",
    "PastNumberOfClaims",
    "AgeOfVehicle",
    "AgeOfPolicyHolder",
    "PoliceReportFiled",
    "WitnessPresent",
    "AgentType",
    "NumberOfSuppliments",
    "AddressChange_Claim",
    "NumberOfCars",
    "BasePolicy",
]

# Mapping dictionary from common snake_case or extraction keys to canonical names
FIELD_NAME_ALIASES: Dict[str, str] = {
    "weekofmonth": "WeekOfMonth",
    "week_of_month": "WeekOfMonth",
    "age": "Age",
    "driver_age": "Age",
    "deductible": "Deductible",
    "driverrating": "DriverRating",
    "driver_rating": "DriverRating",
    "month": "Month",
    "dayofweek": "DayOfWeek",
    "day_of_week": "DayOfWeek",
    "make": "Make",
    "vehicle_make": "Make",
    "accidentarea": "AccidentArea",
    "accident_area": "AccidentArea",
    "sex": "Sex",
    "gender": "Sex",
    "maritalstatus": "MaritalStatus",
    "marital_status": "MaritalStatus",
    "fault": "Fault",
    "at_fault": "Fault",
    "policytype": "PolicyType",
    "policy_type": "PolicyType",
    "vehiclecategory": "VehicleCategory",
    "vehicle_category": "VehicleCategory",
    "vehicle_type": "VehicleCategory",
    "vehicleprice": "VehiclePrice",
    "vehicle_price": "VehiclePrice",
    "days_policy_accident": "Days_Policy_Accident",
    "days_policy_claim": "Days_Policy_Claim",
    "pastnumberofclaims": "PastNumberOfClaims",
    "past_number_of_claims": "PastNumberOfClaims",
    "previous_claims": "PastNumberOfClaims",
    "ageofvehicle": "AgeOfVehicle",
    "age_of_vehicle": "AgeOfVehicle",
    "vehicle_age": "AgeOfVehicle",
    "ageofpolicyholder": "AgeOfPolicyHolder",
    "age_of_policy_holder": "AgeOfPolicyHolder",
    "policereportfiled": "PoliceReportFiled",
    "police_report_filed": "PoliceReportFiled",
    "witnesspresent": "WitnessPresent",
    "witness_present": "WitnessPresent",
    "agenttype": "AgentType",
    "agent_type": "AgentType",
    "numberofsuppliments": "NumberOfSuppliments",
    "number_of_suppliments": "NumberOfSuppliments",
    "addresschange_claim": "AddressChange_Claim",
    "address_change_claim": "AddressChange_Claim",
    "numberofcars": "NumberOfCars",
    "number_of_cars": "NumberOfCars",
    "basepolicy": "BasePolicy",
    "base_policy": "BasePolicy",
}

# Identifiers to discard from model features
IDENTIFIER_KEYS = {
    "policy_number", "policynumber", "rep_number", "repnumber",
    "claim_id", "claimid", "id", "request_id"
}


def normalize_category_value(feature_name: str, val: Any) -> Any:
    """Normalizes capitalization and spacing for common categorical fields."""
    if val is None or not isinstance(val, str):
        return val

    s = val.strip()
    lower_s = s.lower()

    if feature_name == "AccidentArea":
        if lower_s == "urban":
            return "Urban"
        if lower_s == "rural":
            return "Rural"

    elif feature_name == "Sex":
        if lower_s in ["male", "m"]:
            return "Male"
        if lower_s in ["female", "f"]:
            return "Female"

    elif feature_name == "MaritalStatus":
        mapping = {
            "married": "Married",
            "single": "Single",
            "divorced": "Divorced",
            "widow": "Widow"
        }
        return mapping.get(lower_s, s.title())

    elif feature_name == "Fault":
        if "third" in lower_s:
            return "Third Party"
        if "holder" in lower_s or "policy" in lower_s or "insured" in lower_s:
            return "Policy Holder"

    elif feature_name in ["PoliceReportFiled", "WitnessPresent"]:
        if lower_s in ["yes", "y", "true", "1"]:
            return "Yes"
        if lower_s in ["no", "n", "false", "0"]:
            return "No"

    elif feature_name == "AgentType":
        if lower_s == "external":
            return "External"
        if lower_s == "internal":
            return "Internal"

    elif feature_name == "BasePolicy":
        if "collision" in lower_s:
            return "Collision"
        if "liability" in lower_s:
            return "Liability"
        if "peril" in lower_s:
            return "All Perils"

    elif feature_name == "VehicleCategory":
        if "sedan" in lower_s:
            return "Sedan"
        if "sport" in lower_s:
            return "Sport"
        if "utility" in lower_s or "suv" in lower_s or "truck" in lower_s:
            return "Utility"

    return s


def parse_accident_date(date_str: str) -> Dict[str, Any]:
    """Extracts Month, DayOfWeek, and WeekOfMonth from date string if parseable."""
    date_features = {}
    for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%B %d, %Y", "%b %d, %Y"]:
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            date_features["Month"] = dt.strftime("%b")
            date_features["DayOfWeek"] = dt.strftime("%A")
            # Calculate week of month (1 to 5)
            first_day = dt.replace(day=1)
            week_of_month = ((dt.day + first_day.weekday()) // 7) + 1
            date_features["WeekOfMonth"] = min(max(week_of_month, 1), 5)
            break
        except (ValueError, TypeError):
            continue
    return date_features


def map_input_to_model_features(raw_input: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Validates and maps input fields into Part 1 canonical features.

    Returns:
        Tuple of (mapped_features_dict, warnings_list)

    Raises:
        ValueError: If unsupported fields (annual_mileage, etc.) or invalid values are detected.
    """
    # 1. Guard against explicitly unsupported fields
    for unsupp, expl in UNSUPPORTED_FRONTEND_FIELDS.items():
        if unsupp in raw_input:
            raise ValueError(f"Unsupported field '{unsupp}' provided. {expl}")

    mapped: Dict[str, Any] = {}
    warnings: List[str] = []

    # 2. Check for date parsing opportunity if explicit incident date was supplied
    if "accident_date" in raw_input and raw_input["accident_date"]:
        parsed_dates = parse_accident_date(str(raw_input["accident_date"]))
        for k, v in parsed_dates.items():
            if k not in raw_input and k.lower() not in raw_input:
                mapped[k] = v
                warnings.append(f"Derived {k}='{v}' from accident_date '{raw_input['accident_date']}'.")

    # 3. Process fields
    for raw_k, raw_v in raw_input.items():
        if raw_v is None:
            continue

        k_norm = raw_k.strip().lower()

        # Discard pure identifiers with a friendly warning
        if k_norm in IDENTIFIER_KEYS:
            warnings.append(
                f"Field '{raw_k}' is an identifier and was excluded from ML model features."
            )
            continue

        # Find canonical name
        canonical = FIELD_NAME_ALIASES.get(k_norm)
        if not canonical and raw_k in CANONICAL_FEATURES:
            canonical = raw_k

        if not canonical:
            warnings.append(f"Field '{raw_k}' ignored as it is not part of the model schema.")
            continue

        # Type validation & normalization
        if canonical in ["WeekOfMonth", "Age", "Deductible", "DriverRating"]:
            try:
                num_v = float(raw_v)
                if canonical == "Age" and num_v < 0:
                    raise ValueError(f"Age cannot be negative: {num_v}")
                mapped[canonical] = num_v
            except (ValueError, TypeError) as err:
                raise ValueError(f"Invalid numeric value for '{canonical}': '{raw_v}'") from err
        else:
            mapped[canonical] = normalize_category_value(canonical, raw_v)

    return mapped, warnings
