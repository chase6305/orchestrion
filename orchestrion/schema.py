"""Lightweight request schemas for peripheral command validation."""

import copy
import json
import math
from numbers import Real
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Union


def _is_finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, Real)
        and (isinstance(value, int) or math.isfinite(value))
    )


_TYPE_CHECKS = {
    "string": lambda value: isinstance(value, str),
    "number": _is_finite_number,
    "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
    "boolean": lambda value: isinstance(value, bool),
    "object": lambda value: isinstance(value, dict),
    "array": lambda value: isinstance(value, list),
}


def _matches_choice(
    value: Any, choice: Any, is_deferred: Optional[Callable[[Any], bool]]
) -> bool:
    """Compare JSON structures; only explicitly deferred values act as wildcards."""
    if is_deferred is not None and is_deferred(value):
        return True
    if isinstance(value, dict):
        return (
            isinstance(choice, dict)
            and value.keys() == choice.keys()
            and all(
                _matches_choice(child, choice[key], is_deferred)
                for key, child in value.items()
            )
        )
    if isinstance(value, (list, tuple)):
        return (
            isinstance(choice, (list, tuple))
            and len(value) == len(choice)
            and all(
                _matches_choice(child, expected, is_deferred)
                for child, expected in zip(value, choice)
            )
        )
    if isinstance(value, bool) != isinstance(choice, bool):
        return False
    return value == choice


class CommandValidationError(ValueError):
    """A payload violates its declared schema; command inputs fail before I/O."""


class FieldSpec:
    """Describe a field, optionally constraining array items or object children."""

    VALID_TYPES = set(_TYPE_CHECKS)

    def __init__(
        self,
        value_type: str,
        required: bool = False,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
        choices: Optional[Iterable[Any]] = None,
        description: Optional[str] = None,
        *,
        items: Optional["FieldSpec"] = None,
        object_schema: Optional["RequestSchema"] = None,
        min_items: Optional[int] = None,
        max_items: Optional[int] = None,
    ):
        if value_type not in self.VALID_TYPES:
            raise ValueError("Unsupported field type: {}".format(value_type))
        if not isinstance(required, bool):
            raise TypeError("required must be a boolean")
        for name, value in (("minimum", minimum), ("maximum", maximum)):
            if value is not None and not _is_finite_number(value):
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
        if items is not None:
            if not isinstance(items, FieldSpec):
                raise TypeError("items must be FieldSpec or None")
            if value_type != "array":
                raise ValueError("items requires an array field")
        if object_schema is not None:
            if not isinstance(object_schema, RequestSchema):
                raise TypeError("object_schema must be RequestSchema or None")
            if value_type != "object":
                raise ValueError("object_schema requires an object field")
        for name, value in (("min_items", min_items), ("max_items", max_items)):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(
                    "{} must be a nonnegative integer or None".format(name)
                )
        if min_items is not None or max_items is not None:
            if value_type != "array":
                raise ValueError("min_items and max_items require an array field")
            if (
                min_items is not None
                and max_items is not None
                and min_items > max_items
            ):
                raise ValueError("min_items must not exceed max_items")
        choice_list = None if choices is None else list(choices)
        validated_choices = copy.deepcopy(choice_list)
        try:
            json.dumps(validated_choices, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("choices must be JSON-compatible") from exc
        self.value_type = value_type
        self.required = required
        self.minimum = (
            minimum if minimum is None or isinstance(minimum, int) else float(minimum)
        )
        self.maximum = (
            maximum if maximum is None or isinstance(maximum, int) else float(maximum)
        )
        self.choices = None
        self.description = description
        self.items = copy.deepcopy(items)
        self.object_schema = copy.deepcopy(object_schema)
        self.min_items = min_items
        self.max_items = max_items
        if validated_choices is not None:
            # Every candidate is already a member of this enumeration. Check
            # its other constraints without a redundant quadratic membership
            # scan; nested fields still enforce their own choices.
            for choice in validated_choices:
                try:
                    self.validate("choice", choice)
                except CommandValidationError as exc:
                    raise ValueError(
                        "choices must satisfy the field constraints"
                    ) from exc
        self.choices = validated_choices

    def validate(
        self,
        name: str,
        value: Any,
        *,
        is_deferred: Optional[Callable[[Any], bool]] = None,
    ) -> None:
        """Check a value; template validation may defer explicitly marked values."""
        if is_deferred is not None and is_deferred(value):
            return
        valid = _TYPE_CHECKS[self.value_type](value)
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
        if self.choices is not None:
            matches = (
                any(
                    _matches_choice(value, choice, is_deferred)
                    for choice in self.choices
                )
                if self.value_type in {"object", "array"}
                else value in self.choices
            )
            if not matches:
                raise CommandValidationError(
                    "Field {!r} must be one of {}".format(name, self.choices)
                )
        if self.min_items is not None and len(value) < self.min_items:
            raise CommandValidationError(
                "Field {!r} requires at least {} items".format(name, self.min_items)
            )
        if self.max_items is not None and len(value) > self.max_items:
            raise CommandValidationError(
                "Field {!r} allows at most {} items".format(name, self.max_items)
            )
        if self.items is not None:
            for index, item in enumerate(value):
                child_name = "{}[{}]".format(name, index)
                if is_deferred is None:
                    self.items.validate(child_name, item)
                else:
                    self.items.validate(child_name, item, is_deferred=is_deferred)
        if self.object_schema is not None:
            self.object_schema._validate_fields(
                value, prefix=name + ".", is_deferred=is_deferred
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
        if self.items is not None:
            description["items"] = self.items.describe()
        if self.object_schema is not None:
            description["object_schema"] = self.object_schema.describe()
        if self.min_items is not None:
            description["min_items"] = self.min_items
        if self.max_items is not None:
            description["max_items"] = self.max_items
        return description


class RequestSchema:
    """Validate an object payload, including explicitly declared nested fields."""

    def __init__(self, fields: Mapping[str, FieldSpec], allow_extra: bool = False):
        if not isinstance(fields, Mapping):
            raise TypeError("fields must be a mapping")
        if not isinstance(allow_extra, bool):
            raise TypeError("allow_extra must be a boolean")
        if any(
            not isinstance(name, str) or not name or not isinstance(spec, FieldSpec)
            for name, spec in fields.items()
        ):
            raise TypeError("fields must map non-empty strings to FieldSpec values")
        self._fields = copy.deepcopy(dict(fields))
        self._allow_extra = allow_extra

    def validate(self, content: Optional[Dict]) -> Dict:
        """Return an isolated payload, validating the exact copy used by callers."""
        if content is None:
            content = {}
        if not isinstance(content, dict):
            raise CommandValidationError("Command content must be a dictionary")
        snapshot = copy.deepcopy(content)
        if not isinstance(snapshot, dict):
            raise CommandValidationError("Copied command content must be a dictionary")
        self._validate_fields(snapshot)
        return snapshot

    def validate_template(
        self, content: Dict, *, is_deferred: Callable[[Any], bool]
    ) -> None:
        """Check known fields while deferring values selected by the predicate.

        Literal containers still enforce required fields, types and lengths.
        Composite choices must match their known parts; deferred values are rechecked
        after binding.
        This does not return executable input: resolve bindings and call validate()
        before dispatch. The schema has no dependency on any binding representation.
        """
        if not callable(is_deferred):
            raise TypeError("is_deferred must be callable")
        if not isinstance(content, dict):
            raise CommandValidationError("Command content must be a dictionary")
        self._validate_fields(content, is_deferred=is_deferred)

    def _validate_fields(
        self,
        content: Dict,
        prefix: str = "",
        *,
        is_deferred: Optional[Callable[[Any], bool]] = None,
    ) -> None:
        if any(not isinstance(name, str) for name in content):
            raise CommandValidationError(
                "{}field names must be strings".format(prefix or "Command ")
            )
        if not self._allow_extra:
            unknown = content.keys() - self._fields.keys()
            if unknown:
                raise CommandValidationError(
                    "Unknown command fields: {}".format(
                        ", ".join(prefix + name for name in sorted(unknown))
                    )
                )
        for name, spec in self._fields.items():
            if name not in content:
                if spec.required:
                    raise CommandValidationError(
                        "Missing required field: {!r}".format(prefix + name)
                    )
                continue
            if is_deferred is None:
                spec.validate(prefix + name, content[name])
            else:
                spec.validate(prefix + name, content[name], is_deferred=is_deferred)

    def validate_path(self, path: Iterable[Union[str, int]]) -> None:
        """Reject paths impossible under this schema; optional/unknown values defer.

        This proves structural compatibility only. Optional fields and array
        elements can still be absent at runtime, and open objects are unresolved.
        """
        if isinstance(path, (str, bytes)):
            raise TypeError("path must be a sequence of keys or indices")
        parts = tuple(path)
        for part in parts:
            if isinstance(part, bool) or not isinstance(part, (str, int)):
                raise TypeError("path requires string keys or integer indices")
            if isinstance(part, int) and part < 0:
                raise ValueError("path indices must be nonnegative")
        current_schema = self
        current_field = None
        for part in parts:
            value_type = "object" if current_field is None else current_field.value_type
            if isinstance(part, str):
                if value_type != "object":
                    raise CommandValidationError("output path requires an object")
                if current_schema is None:
                    return
                if part not in current_schema._fields:
                    if current_schema._allow_extra:
                        return
                    raise CommandValidationError("unknown output field: " + part)
                current_field = current_schema._fields[part]
            else:
                if value_type != "array":
                    raise CommandValidationError("output path requires an array")
                if (
                    current_field.max_items is not None
                    and part >= current_field.max_items
                ):
                    raise CommandValidationError(
                        "output index exceeds declared max_items"
                    )
                current_field = current_field.items
                if current_field is None:
                    return
            current_schema = current_field.object_schema

    def describe(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "allow_extra": self._allow_extra,
            "fields": {name: spec.describe() for name, spec in self._fields.items()},
        }
