"""ament_python package setup for medortrace_ros (ROS 2 Humble / Jazzy)."""

from glob import glob

from setuptools import find_packages, setup

PACKAGE = "medortrace_ros"

setup(
    name=PACKAGE,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{PACKAGE}"]),
        (f"share/{PACKAGE}", ["package.xml", "README.md"]),
        (f"share/{PACKAGE}/launch", glob("launch/*.launch.py")),
        (f"share/{PACKAGE}/config", glob("config/*.yaml") + glob("config/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="MED-OR-TRACE maintainers",
    maintainer_email="medortrace@example.org",
    description="ROS 2 interface of the MED-OR-TRACE object-chain verifier",
    license="Apache-2.0",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            f"autonomy_node = {PACKAGE}.autonomy_node:main",
            f"sim_bridge_node = {PACKAGE}.sim_bridge_node:main",
            f"workflow_gateway_node = {PACKAGE}.workflow_gateway_node:main",
            f"operator_console_node = {PACKAGE}.operator_console_node:main",
        ],
    },
)
