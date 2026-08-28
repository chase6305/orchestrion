"""Lightweight request schemas for peripheral command validation."""

import copy
import json
import math
from numbers import Real
from typing import Any, Dict, Iterable, Mapping, Optional


class CommandValidationError(ValueError):
    """Raised before device I/O when a command payload violates its schema."""


class FieldSpec:
    """Describe one top-level command field."""

    VALID_TYPES = {"string", "number", "integer", "boolean", "object", "array"}

    def __init__(
        self,
        value_type: str,
        required: bool = False,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
        choices: Optional[Iterable[Any]] = None,
        description: Optional[str] = None,
    ):
        if value_type not in self.VALID_TYPES:
            raise ValueError("Unsupported field type: {}".format(value_type))
        if not isinstance(required, bool):
            raise TypeError("required must be a boolean")
        for name, value in (("minimum", minimum), ("maximum", maximum)):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
            ):
                raise ValueError("{} must be finite or None".format(name))
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError("minimum must not exceed maximum")
        if (minimum is not None or maximum is not None) and value_type not in {
            "number",
            "integer",
        }:
            raise ValueError("minimum and maximum require a numeric field type")
        if description is not None and not isinstance(description, str):
            raise TypeError("description must be a string or None")
        choice_list = None if choices is None else list(choices)
        try:
            json.dumps(choice_list, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("choices must be JSON-compatible") from exc
        self.value_type = value_type
        self.required = required
        self.minimum = None if minimum is None else float(minimum)
        self.maximum = None if maximum is None else float(maximum)
        self.choices = copy.deepcopy(choice_list)
        self.description = description
        if self.choices is not None:
            for choice in self.choices:
                try:
                    self.validate("choice", choice)
                except CommandValidationError as exc:
                    raise ValueError(
                        "choices must satisfy the field constraints"
                    ) from exc

    def validate(self, name: str, value: Any) -> None:
        valid = {
            "string": lambda item: isinstance(item, str),
            "number": lambda item: isinstance(item, Real)
            and not isinstance(item, bool)
            and math.isfinite(item),
            "integer": lambda item: isinstance(item, int)
            and not isinstance(item, bool),
            "boolean": lambda item: isinstance(item, bool),
            "object": lambda item: isinstance(item, dict),
            "array": lambda item: isinstance(item, list),
        }[self.value_type](value)
        if not valid:
            raise CommandValidationError(
                "Field {!r} must be {}".format(name, self.value_type)
            )
        if self.minimum is not None and value < self.minimum:
            raise CommandValidationError(
                "Field {!r} must be at least {}".format(name, self.minimum)
            )
        if self.maximum is not None and value > self.maximum:
            raise CommandValidationError(
                "Field {!r} must be at most {}".format(name, self.maximum)
            )
        if self.choices is not None and value not in self.choices:
            raise CommandValidationError(
                "Field {!r} must be one of {}".format(name, self.choices)
            )

    def describe(self) -> Dict[str, Any]:
        description = {"type": self.value_type, "required": self.required}
        if self.minimum is not None:
            description["minimum"] = self.minimum
        if self.maximum is not None:
            description["maximum"] = self.maximum
        if self.choices is not None:
            description["choices"] = copy.deepcopy(self.choices)
        if self.description is not None:
            description["description"] = self.description
        return description


class RequestSchema:
    """Validate a flat dictionary payload before invoking a peripheral SDK."""

    def __init__(
        self, fields: Mapping[str, FieldSpec], allow_extra: bool = False
    ):
        if not isinstance(fields, Mapping):
            raise TypeError("fields must be a mapping")
        if not isinstance(allow_extra, bool):
            raise TypeError("allow_extra must be a boolean")
        if any(
            not isinstance(name, str) or not name or not isinstance(spec, FieldSpec)
            for name, spec in fields.items()
        ):
            raise TypeError("fields must map non-empty strings to FieldSpec values")
        self._fields = dict(fields)
        self._allow_extra = allow_extra

    def validate(self, content: Optional[Dict]) -> Dict:
        if content is None:
            content = {}
        if not isinstance(content, dict):
            raise CommandValidationError("Command content must be a dictionary")
        if any(not isinstance(name, str) for name in content):
            raise CommandValidationError("Command field names must be strings")
        unknown = set(content) - set(self._fields)
        if unknown and not self._allow_extra:
            raise CommandValidationError(
                "Unknown command fields: {}".format(", ".join(sorted(unknown)))
            )
        for name, spec in self._fields.items():
            if name not in content:
                if spec.required:
                    raise CommandValidationError(
                        "Missing required field: {!r}".format(name)
                    )
                continue
            spec.validate(name, content[name])
        return copy.deepcopy(content)

    def describe(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "allow_extra": self._allow_extra,
            "fields": {
                name: spec.describe() for name, spec in self._fields.items()
            },
        }
