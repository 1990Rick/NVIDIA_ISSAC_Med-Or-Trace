"""Small rclpy helpers shared by the nodes (ROS imports are lazy)."""

from __future__ import annotations


def param(node, name: str, default):
    """Declare-if-needed and read a parameter.

    Parameters are declared with dynamic typing so that ``duration:=60`` (an integer on the
    command line / in a launch substitution) does not fail against a float default; callers cast.
    """
    if not node.has_parameter(name):
        from rcl_interfaces.msg import ParameterDescriptor

        node.declare_parameter(name, default, ParameterDescriptor(dynamic_typing=True))
    v = node.get_parameter(name).value
    return default if v is None else v
