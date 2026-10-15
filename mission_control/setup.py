from setuptools import setup

package_name = "mission_control"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", ["config/mission.yaml"]),
        ("share/" + package_name + "/launch", ["launch/mission.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="bichy",
    maintainer_email="152578797+Ancientree-Bichy@users.noreply.github.com",
    description="RoboCup rescue mission layer",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "goal_selector = mission_control.goal_selector:main",
            "end_detector = mission_control.end_detector:main",
            "mission_fsm = mission_control.mission_fsm:main",
            "recorded_path_manager = mission_control.recorded_path_manager:main",
            "path_source_mux = mission_control.path_source_mux:main",
        ],
    },
)
