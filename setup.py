from setuptools import setup, find_packages

setup(
    name="ext-to-ego",
    version="0.1.0",
    description="Add your description here",
    readme="README.md",
    python_requires=">=3.12",
    install_requires=[
        "matplotlib>=3.10.9",
        "numpy>=2.4.4",
        "opencv-python>=4.13.0.92",
        "pyyaml>=6.0.3",
        "scipy>=1.17.1",
        "torch>=2.0.0",
        "ultralytics>=8.4.47",
        "zarr>=3.0.0",
        "pyrealsense2>=2.57.7.10387",
    ],
    packages=find_packages(),
)
