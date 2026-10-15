from setuptools import find_packages, setup


package_name = "qrc_ros2_perception"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (
            f"share/{package_name}/launch",
            ["launch/qrc_sensing.launch.py", "launch/qrc_result_panel.launch.py"],
        ),
        (
            f"share/{package_name}/configs",
            [
                "configs/realsense.yaml",
                "configs/pyrealsense2.yaml",
                "configs/playback.yaml",
                "configs/robotdog.example.yaml",
                "configs/result_panel.yaml",
            ],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=False,
    maintainer="Ancientree-Bichy",
    maintainer_email="152578797+Ancientree-Bichy@users.noreply.github.com",
    description="ROS2 Humble sensing bridge for the QRC inspection pipeline.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "qrc-sensing-node = qrc_ros2_perception.node:main",
            "qrc-result-panel = qrc_ros2_perception.result_panel_node:main",
            "qrc-test-image-publisher = qrc_ros2_perception.test_image_publisher:main",
        ],
    },
)
