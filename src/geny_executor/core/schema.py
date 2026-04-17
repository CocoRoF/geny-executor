"""Self-describing configuration schema system.

Allows stages and strategies to expose their configurable parameters
as structured metadata, enabling dynamic UI form generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class ConfigField:
    """Single configurable parameter definition."""

    name: str
    type: str  # "string" | "integer" | "number" | "boolean" | "select" | "array" | "object"
    label: str
    description: str = ""
    default: Any = None
    required: bool = False

    # Numeric constraints
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None

    # String constraints
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None

    # Select options: [{"value": "...", "label": "..."}]
    options: Optional[List[Dict[str, str]]] = None

    # Array item type
    item_type: Optional[str] = None

    # Nested object properties
    properties: Optional[List[ConfigField]] = None

    # UI rendering hints
    ui_widget: Optional[str] = None  # "textarea" | "code" | "password" | "slider" | "toggle"
    ui_group: Optional[str] = None
    ui_order: int = 0

    # Conditional visibility: {"field_name": expected_value}
    visible_when: Optional[Dict[str, Any]] = None

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema compatible dict."""
        schema: Dict[str, Any] = {}

        # Map type
        type_map = {
            "string": "string",
            "integer": "integer",
            "number": "number",
            "boolean": "boolean",
            "select": "string",
            "array": "array",
            "object": "object",
        }
        schema["type"] = type_map.get(self.type, "string")

        # Metadata
        schema["title"] = self.label
        if self.description:
            schema["description"] = self.description
        if self.default is not None:
            schema["default"] = self.default

        # Numeric
        if self.min_value is not None:
            schema["minimum"] = self.min_value
        if self.max_value is not None:
            schema["maximum"] = self.max_value

        # String
        if self.min_length is not None:
            schema["minLength"] = self.min_length
        if self.max_length is not None:
            schema["maxLength"] = self.max_length
        if self.pattern is not None:
            schema["pattern"] = self.pattern

        # Enum (select)
        if self.options:
            schema["enum"] = [o["value"] for o in self.options]
            schema["enumLabels"] = {o["value"]: o["label"] for o in self.options}

        # Array
        if self.type == "array" and self.item_type:
            if self.properties:
                schema["items"] = ConfigSchema(
                    name=f"{self.name}_item",
                    fields=self.properties,
                ).to_json_schema()
            else:
                schema["items"] = {"type": type_map.get(self.item_type, "string")}

        # Object
        if self.type == "object" and self.properties:
            sub_schema = ConfigSchema(name=self.name, fields=self.properties)
            sub = sub_schema.to_json_schema()
            schema["properties"] = sub.get("properties", {})
            if sub.get("required"):
                schema["required"] = sub["required"]

        # UI extensions (non-standard, prefixed)
        if self.ui_widget:
            schema["x-ui-widget"] = self.ui_widget
        if self.ui_group:
            schema["x-ui-group"] = self.ui_group
        if self.ui_order:
            schema["x-ui-order"] = self.ui_order
        if self.visible_when:
            schema["x-visible-when"] = self.visible_when

        return schema


@dataclass
class ConfigSchema:
    """Complete configuration schema for a Stage or Strategy."""

    name: str
    version: str = "1.0"
    fields: List[ConfigField] = field(default_factory=list)
    groups: List[Dict[str, str]] = field(default_factory=list)

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to full JSON Schema object."""
        properties: Dict[str, Any] = {}
        required: List[str] = []

        for f in sorted(self.fields, key=lambda x: x.ui_order):
            properties[f.name] = f.to_json_schema()
            if f.required:
                required.append(f.name)

        schema: Dict[str, Any] = {
            "type": "object",
            "title": self.name,
            "properties": properties,
        }
        if required:
            schema["required"] = required
        if self.groups:
            schema["x-ui-groups"] = self.groups

        return schema

    def validate(self, data: Dict[str, Any]) -> List[str]:
        """Validate configuration data against this schema. Returns error messages."""
        errors: List[str] = []

        for f in self.fields:
            value = data.get(f.name)

            # Required check
            if f.required and value is None:
                errors.append(f"Required field '{f.name}' is missing")
                continue

            if value is None:
                continue

            # Type check
            errors.extend(self._validate_field(f, value))

        return errors

    def _validate_field(self, f: ConfigField, value: Any) -> List[str]:
        """Validate a single field value."""
        errors: List[str] = []

        # Type validation
        type_checks = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "select": str,
            "array": list,
            "object": dict,
        }
        expected = type_checks.get(f.type)
        if expected and not isinstance(value, expected):
            errors.append(f"Field '{f.name}' expected {f.type}, got {type(value).__name__}")
            return errors

        # Numeric bounds
        if f.min_value is not None and isinstance(value, (int, float)):
            if value < f.min_value:
                errors.append(f"Field '{f.name}': {value} < minimum {f.min_value}")
        if f.max_value is not None and isinstance(value, (int, float)):
            if value > f.max_value:
                errors.append(f"Field '{f.name}': {value} > maximum {f.max_value}")

        # String length
        if isinstance(value, str):
            if f.min_length is not None and len(value) < f.min_length:
                errors.append(f"Field '{f.name}': length {len(value)} < minimum {f.min_length}")
            if f.max_length is not None and len(value) > f.max_length:
                errors.append(f"Field '{f.name}': length {len(value)} > maximum {f.max_length}")

        # Enum (select)
        if f.options and isinstance(value, str):
            valid_values = [o["value"] for o in f.options]
            if value not in valid_values:
                errors.append(f"Field '{f.name}': '{value}' not in {valid_values}")

        return errors

    def apply_defaults(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Return a new dict with missing fields filled by their defaults."""
        result = dict(data)
        for f in self.fields:
            if f.name not in result and f.default is not None:
                result[f.name] = f.default
        return result
