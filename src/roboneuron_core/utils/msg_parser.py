"""ROS message schema parser and indexer utilities."""

from __future__ import annotations

import json
import os
import re
from typing import Any

# Mapping from ROS primitive types to JSON-compatible Python type names.
ROS_TO_PYTHON_TYPE: dict[str, str] = {
    "float64": "float",
    "float32": "float",
    "int64": "int",
    "int32": "int",
    "int16": "int",
    "int8": "int",
    "uint64": "int",
    "uint32": "int",
    "uint16": "int",
    "uint8": "int",
    "string": "str",
    "bool": "bool",
    "char": "int",
    "byte": "int",
}


def build_msg_index(base_dirs: list[str]) -> dict[str, str]:
    """Traverse .msg directories and construct a type index."""
    type_index: dict[str, str] = {}
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            continue
        pkg_name = os.path.basename(os.path.dirname(base_dir))
        for file_name in os.listdir(base_dir):
            if not file_name.endswith(".msg"):
                continue
            msg_name = file_name.removesuffix(".msg")
            full_msg_name = f"{pkg_name}/{msg_name}"
            file_path = os.path.join(base_dir, file_name)
            type_index[msg_name] = file_path
            type_index[full_msg_name] = file_path
    return type_index


def parse_ros_type(type_str: str) -> tuple[str, bool, int | None]:
    """Parse a ROS field type string into base type and array metadata."""
    match = re.match(r"^(.+?)(?:\[(\d*)\])?$", type_str)
    if not match:
        raise ValueError(f"Invalid type string: {type_str}")
    base_type, size_str = match.groups()
    is_array = size_str is not None
    size = int(size_str) if size_str else None
    return base_type, is_array, size


def parse_msg_file(msg_path: str) -> list[tuple[str, str]]:
    """Parse a ROS .msg file and extract (type, field_name) definitions."""
    fields: list[tuple[str, str]] = []
    with open(msg_path, encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" in line:
                continue
            tokens = line.split()
            if len(tokens) < 2:
                continue
            field_type, field_name = tokens[:2]
            fields.append((field_type, field_name))
    return fields


def resolve_type_structure(
    type_name: str,
    type_index: dict[str, str],
    cache: dict[str, Any],
) -> Any:
    """Recursively resolve a ROS type into a JSON-compatible schema."""
    if type_name in ROS_TO_PYTHON_TYPE:
        return ROS_TO_PYTHON_TYPE[type_name]
    if type_name in cache:
        return cache[type_name]
    if type_name not in type_index:
        raise ValueError(f"Type '{type_name}' not found in index.")

    msg_path = type_index[type_name]
    fields = parse_msg_file(msg_path)

    current_struct: dict[str, Any] = {}
    cache[type_name] = current_struct

    for field_type_raw, field_name in fields:
        base_type, is_array, array_size = parse_ros_type(field_type_raw)
        resolved_element_type = resolve_type_structure(base_type, type_index, cache)

        if is_array:
            array_representation: list[Any] = [resolved_element_type]
            if array_size is not None:
                array_representation.append(array_size)
            current_struct[field_name] = array_representation
        else:
            current_struct[field_name] = resolved_element_type

    return current_struct


class ROSMsgIndexer:
    """Indexer for parsing and exporting ROS .msg type schemas."""

    def __init__(self, base_path: str, target_pkgs: list[str]) -> None:
        base_msg_dir = os.path.join(base_path, "msg")
        base_pkg_name = os.path.basename(os.path.normpath(base_path))
        base_dirs: list[str] = []

        for pkg in target_pkgs:
            nested_pkg_msg_dir = os.path.join(base_path, pkg, "msg")
            if os.path.isdir(nested_pkg_msg_dir):
                base_dirs.append(nested_pkg_msg_dir)
                continue

            # Allow passing a package root directly (e.g., base_path="ros/custom_msgs")
            if os.path.isdir(base_msg_dir) and pkg == base_pkg_name:
                base_dirs.append(base_msg_dir)

        self.base_dirs = base_dirs
        self.type_index = build_msg_index(self.base_dirs)
        self.cache: dict[str, Any] = {}

    def generate_full_index(self, output_path: str) -> None:
        """Generate and export all known message type structures."""
        full_struct: dict[str, Any] = {}
        for type_name in self.type_index:
            full_struct[type_name] = resolve_type_structure(type_name, self.type_index, self.cache)

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(full_struct, file, indent=2)
        print(f"ROS2 msg index saved to: {output_path}")

    def get_type_structure(self, type_name: str) -> Any:
        """Return the resolved structure for a single message type."""
        if type_name in self.cache:
            return self.cache[type_name]
        return resolve_type_structure(type_name, self.type_index, self.cache)


if __name__ == "__main__":
    BASE_PATH = "ros/custom_msgs"
    TARGET_PKGS = ["custom_msgs"]

    indexer = ROSMsgIndexer(BASE_PATH, TARGET_PKGS)
    indexer.generate_full_index("ros/custom_msgs/ros_msg_index.json")

    eecommand_struct = indexer.get_type_structure("custom_msgs/EECommand")
    print(json.dumps(eecommand_struct, indent=2))
