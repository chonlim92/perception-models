from setuptools import setup, find_packages

setup(
    name="perception-models",
    version="1.0.0",
    description="State-of-the-art perception models for autonomous driving",
    author="Autonomous Driving Perception Team",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "PyYAML>=6.0",
        "einops>=0.6.0",
        "timm>=0.9.0",
        "tqdm>=4.65.0",
        "pyquaternion>=0.9.9",
        "shapely>=2.0.0",
        "nuscenes-devkit>=1.1.10",
    ],
    extras_require={
        "tensorflow": ["tensorflow>=2.13.0"],
        "visualization": ["open3d>=0.17.0", "matplotlib>=3.7.0"],
        "scenario": [
            "transformers>=4.30.0",
            "sentence-transformers>=2.2.0",
            "sqlalchemy>=2.0.0",
            "pydantic>=2.0.0",
            "hdbscan>=0.8.33",
        ],
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0"],
    },
)
