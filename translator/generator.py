#!/usr/bin/env python3
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from typing import List, Optional
import argparse
import msg_parser as mp

# --- Global Defaults ---
DEFAULT_TEMPLATE_DIR = "template"
DEFAULT_OUTPUT_DIR = Path("mcptool_lib")
DEFAULT_ROS_MSG_BASE_PATH = "ros2_msg"
DEFAULT_ROS_TARGET_PKGS = [
    "geometry_msgs",
    "std_msgs",
    "builtin_interfaces",
    "custom_msgs"
]


class MCPToolGenerator:
    """
    Generates Python-based MCP tools using Jinja2 templates and ROS .msg type introspection.
    Now optimized for Native ROS 2 Nodes (rclpy).
    """

    def __init__(
        self,
        template_dir: str = DEFAULT_TEMPLATE_DIR,
        ros_msg_base_path: str = DEFAULT_ROS_MSG_BASE_PATH,
        ros_target_pkgs: Optional[List[str]] = None,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
    ):
        if ros_target_pkgs is None:
            ros_target_pkgs = DEFAULT_ROS_TARGET_PKGS

        self.template_env = Environment(loader=FileSystemLoader(template_dir))
        self.template = self.template_env.get_template("mcptool_template.jinja2")
        self.indexer = mp.ROSMsgIndexer(ros_msg_base_path, ros_target_pkgs)
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_mcp_tool(
        self,
        topic_name: str,
        msg_type_name: str,
        output_filename: Optional[str] = None
    ):
        """
        Generate an MCP tool file for the specified ROS message type.
        """
        try:
            # 1. Parse Package and Message Name
            msg_package = "unknown_pkg"
            msg_name_short_form = msg_type_name

            if "/" in msg_type_name:
                parts = msg_type_name.split("/")
                msg_package = parts[0]
                msg_name_short_form = parts[-1]
            else:
                # Try to resolve using indexer if only ShortName provided
                resolved_path = self.indexer.type_index.get(msg_type_name)
                if resolved_path:
                    parts = Path(resolved_path).parts
                    # Assumes structure: .../package_name/msg/MessageName.msg
                    if "msg" in parts:
                        idx = parts.index("msg")
                        if idx > 0:
                            msg_package = parts[idx - 1]

            # 2. Get Message Structure (Fields)
            msg_structure = self.indexer.get_type_structure(msg_type_name)

            if not msg_structure or "error" in msg_structure:
                raise ValueError(
                    f"Could not retrieve valid structure for message type '{msg_type_name}'. "
                    f"Error: {msg_structure.get('error', 'Unknown')}"
                )

            # 3. Render Template
            rendered_content = self.template.render(
                topic_name=topic_name,
                msg_name=msg_name_short_form,
                msg_package=msg_package,
                msg_structure=msg_structure,
            )

            # 4. Save File
            if output_filename is None:
                output_filename = f"{msg_name_short_form.lower()}_mcptool.py"

            output_path = self.output_dir / output_filename

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(rendered_content)

            print(f"✅ Generated ROS 2 Native MCP Tool: {output_path}")

        except ValueError as e:
            print(f"❌ Error generating MCP tool for '{msg_type_name}': {e}")
        except Exception as e:
            print(f"❌ An unexpected error occurred: {e}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate MCP tool file for a ROS message type using Jinja2 template."
    )
    p.add_argument("topic", help="ROS topic name to use in generated tool (e.g. ee_command)")
    p.add_argument("msg", help="ROS message type (short name or package/Message, e.g. EECommand or std_msgs/String)")
    p.add_argument(
        "--output",
        "-o",
        help="Output filename (placed under output dir). Default: <message>_mcptool.py",
    )
    p.add_argument(
        "--template-dir",
        default=DEFAULT_TEMPLATE_DIR,
        help=f"Jinja2 template directory (default: {DEFAULT_TEMPLATE_DIR})",
    )
    p.add_argument(
        "--ros-msg-base",
        default=DEFAULT_ROS_MSG_BASE_PATH,
        help=f"Base path for ROS .msg files (default: {DEFAULT_ROS_MSG_BASE_PATH})",
    )
    p.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory to write generated files into (default: {DEFAULT_OUTPUT_DIR})",
    )
    p.add_argument(
        "--ros-pkgs",
        nargs="+",
        default=DEFAULT_ROS_TARGET_PKGS,
        help="List of ROS packages to index (default includes geometry_msgs, std_msgs, builtin_interfaces, diy_msgs)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    template_dir = args.template_dir
    ros_msg_base = args.ros_msg_base
    ros_pkgs = args.ros_pkgs
    output_dir = Path(args.output_dir)

    # basic checks
    if not Path(template_dir).exists():
        print(f"Error: Template directory '{template_dir}' not found.")
        exit(1)

    generator = MCPToolGenerator(
        template_dir=template_dir,
        ros_msg_base_path=ros_msg_base,
        ros_target_pkgs=ros_pkgs,
        output_dir=output_dir,
    )

    print("\n--- Generating MCP Tool (Native ROS 2) ---")
    generator.generate_mcp_tool(args.topic, args.msg, output_filename=args.output)
    print("\n--- Generation Complete ---")
