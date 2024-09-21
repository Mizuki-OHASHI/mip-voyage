import json
from typing import Callable, Tuple, Optional

ValidateResult = Tuple[bool, Optional[dict]]


def validate_data(data: dict, structure: dict) -> ValidateResult:
    """Validate JSON data.

    Args:
        data (dict): JSON data
        structure (dict): JSON data structure
    Returns:
        bool: True if data is valid, False otherwise
    Examples:
        >>> validate_data({"name": "John", "age": 30, "emails": ["john@example.com"]}, {"name": str, "age": int, "emails": [str]})
        True, None
        >>> validate_data({"name": "John", "age": "30"}, {"name": str, "age": int})
        False, { path: "age", value: "30", expected: int }
    """

    def validate(data, structure) -> bool:
        if isinstance(structure, dict):
            for key, value in structure.items():
                if key not in data:
                    return False, {
                        "path": key,
                        "value": None,
                        "expected": structure_to_str(value),
                    }
                result, info = validate(data[key], value)
                if not result:
                    return False, {"path": key + "." + info["path"], **info}
        elif isinstance(structure, list):
            for item in data:
                result, info = validate(item, structure[0])
                if not result:
                    return False, info
        else:
            val = (int, float) if structure == float else structure
            result = isinstance(data, val)
            return result, (
                None
                if result
                else {
                    "path": "",
                    "value": data,
                    "expected": structure_to_str(structure),
                }
            )
        return True, None

    return validate(data, structure)


def structure_to_str(structure: dict) -> dict:
    """convert all types in structure to string"""

    def to_str(structure):
        if isinstance(structure, dict):
            return {key: to_str(value) for key, value in structure.items()}
        elif isinstance(structure, list):
            return [to_str(structure[0])]
        else:
            return structure.__name__

    return to_str(structure)


def build_validate_data(structure: dict) -> Callable[[dict], ValidateResult]:
    return lambda data: validate_data(data, structure)


def parse_json(
    data_str: str, validator: Callable[[dict], ValidateResult] = validate_data
) -> Tuple[bool, dict]:
    """Parse JSON data and validate it.

    Args:
        data_str (str): JSON data in string format
        validator (Callable[[dict, dict], bool], optional): Function to validate JSON data. Defaults to validate_data.
    Returns:
       (bool, dict): (True, data) if data is valid, (False, error information) otherwise
    """

    data = json.loads(data_str)
    ok, info = validator(data)
    return ok, data if ok else info


def read_file(file_path: str) -> str:
    with open(file_path, "r") as file:
        return file.read()


if __name__ == "__main__":
    data_str = read_file("../asset/data.json")
    print("Data in string format:", data_str, sep="\n")

    # normal usage
    structure = {"id": str, "profile": {"name": str, "age": int, "emails": [str]}}
    validator = build_validate_data(structure)
    data = parse_json(data_str, validator)
    print("Data in dict format:", data, sep="\n")

    def run_negative_test_case(structure: dict):
        validator = build_validate_data(structure)
        try:
            _ = parse_json(data_str, validator)
            raise SystemError("test failed: parse_json should raise ValueError")
        except ValueError as e:
            print("test passed: parse_json raised ValueError:", e)

    negative_test_cases = [
        {"id": str, "profile": {"name": str, "age": int, "emails": str}},
        {"id": str, "profile": {"name": str, "age": int, "emails": [int]}},
        {"id": str, "name": str, "age": int, "emails": [str]},
    ]
    for structure in negative_test_cases:
        run_negative_test_case(structure)
