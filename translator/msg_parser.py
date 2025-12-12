import os
import json
import re

# Mapping from ROS primitive types to JSON-compatible Python types
ROS_TO_PYTHON_TYPE = {
    "float64": "float", "float32": "float",
    "int64": "int", "int32": "int", "int16": "int", "int8": "int",
    "uint64": "int", "uint32": "int", "uint16": "int", "uint8": "int",
    "string": "str", "bool": "bool",
    "char": "int",
    "byte": "int"
}


def build_msg_index(base_dirs):
    """
    Traverse .msg directories and construct a type index.

    Supports both 'MsgName' and 'pkg_name/MsgName' keys.

    Args:
        base_dirs (List[str]): List of paths to 'msg' directories of ROS packages.

    Returns:
        Dict[str, str]: Mapping from message type name to file path.
    """
    type_index = {}
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            continue
        pkg_name = os.path.basename(os.path.dirname(base_dir))
        for file in os.listdir(base_dir):
            if file.endswith(".msg"):
                msg_name = file.replace(".msg", "")
                full_msg_name = f"{pkg_name}/{msg_name}"
                file_path = os.path.join(base_dir, file)
                type_index[msg_name] = file_path
                type_index[full_msg_name] = file_path
    return type_index


def parse_ros_type(type_str):
    """
    Parse a field type string into base type and array metadata.

    Args:
        type_str (str): ROS-style field type string (e.g. "float32[3]").

    Returns:
        Tuple[str, bool, Optional[int]]: base type, is_array, and array size (if fixed).
    """
    match = re.match(r"^(.+?)(?:\[(\d*)\])?$", type_str)
    if not match:
        raise ValueError(f"Invalid type string: {type_str}")
    base_type, size_str = match.groups()
    is_array = size_str is not None
    size = int(size_str) if size_str else None
    return base_type, is_array, size


def parse_msg_file(msg_path):
    """
    Parse a ROS .msg file and extract field definitions.

    Args:
        msg_path (str): Path to the .msg file.

    Returns:
        List[Tuple[str, str]]: List of (type, field_name) tuples.
    """
    fields = []
    with open(msg_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" in line:
                continue
            tokens = line.split()
            if len(tokens) < 2:
                continue
            type_, name = tokens[:2]
            fields.append((type_, name))
    return fields


def resolve_type_structure(type_name, type_index, cache):
    """
    Recursively resolve the structure of a ROS message type.

    Supports nested sub-messages and fixed-length or dynamic arrays.

    Args:
        type_name (str): Message type name (e.g. 'geometry_msgs/Twist').
        type_index (Dict[str, str]): Index of message names to file paths.
        cache (Dict[str, Any]): Cache for resolved message types.

    Returns:
        Union[Dict[str, Any], str, List]: JSON-compatible type structure.
    """
    if type_name in ROS_TO_PYTHON_TYPE:
        return ROS_TO_PYTHON_TYPE[type_name]
    if type_name in cache:
        return cache[type_name]
    if type_name not in type_index:
        raise ValueError(f"Type '{type_name}' not found in index.")

    msg_path = type_index[type_name]
    fields = parse_msg_file(msg_path)

    current_struct = {}
    cache[type_name] = current_struct

    for field_type_raw, field_name in fields:
        base_type, is_array, array_size = parse_ros_type(field_type_raw)
        resolved_element_type = resolve_type_structure(base_type, type_index, cache)

        if is_array:
            array_representation = [resolved_element_type]
            if array_size is not None:
                array_representation.append(array_size)
            current_struct[field_name] = array_representation
        else:
            current_struct[field_name] = resolved_element_type

    return current_struct


class ROSMsgIndexer:
    """
    ROS message indexer for parsing and serializing ROS2 .msg types into JSON-compatible schemas.

    Provides message introspection and structure export for type-safe data manipulation.
    """

    def __init__(self, base_path, target_pkgs):
        """
        Args:
            base_path (str): Base directory containing ROS2 packages.
            target_pkgs (List[str]): List of package names to index (e.g., ['std_msgs']).
        """
        self.base_dirs = [os.path.join(base_path, pkg, "msg") for pkg in target_pkgs]
        self.type_index = build_msg_index(self.base_dirs)
        self.cache = {}

    def generate_full_index(self, output_path):
        """
        Generate and export all known message types as a unified JSON schema file.

        Args:
            output_path (str): Path to save the generated JSON index file.
        """
        full_struct = {}
        for type_name in self.type_index:
            full_struct[type_name] = resolve_type_structure(type_name, self.type_index, self.cache)

        with open(output_path, "w") as f:
            json.dump(full_struct, f, indent=2)
        print(f"✅ ROS2 msg index saved to: {output_path}")

    def get_type_structure(self, type_name):
        """
        Retrieve the structure of a single message type.

        Args:
            type_name (str): Full or partial message type name.

        Returns:
            Dict[str, Any]: JSON-compatible structure for the given type.
        """
        if type_name in self.cache:
            return self.cache[type_name]
        return resolve_type_structure(type_name, self.type_index, self.cache)


if __name__ == "__main__":
    base_path = "ros2_msg"
    target_pkgs = ["geometry_msgs", "std_msgs", "builtin_interfaces"]

    indexer = ROSMsgIndexer(base_path, target_pkgs)

    indexer.generate_full_index("ros2_msg/ros2_msg_index.json")

    twist_struct = indexer.get_type_structure("geometry_msgs/Twist")
    print(json.dumps(twist_struct, indent=2))
