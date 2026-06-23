"""
Setup configuration for the anonymous WSSS histopathology segmentation code.
"""
from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="anonymous-wsss-histopathology",
    version="1.0.0",
    author="Anonymous",
    description="Weakly supervised semantic segmentation for histopathology images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://anonymous.4open.science/",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=23.0.0",
            "flake8>=6.0.0",
            "pytest>=7.3.0",
            "jupyter>=1.0.0",
        ],
        "crf": [
            "pydensecrf @ git+https://github.com/lucasb-eyer/pydensecrf.git#egg=pydensecrf",
        ],
    },
    entry_points={
        "console_scripts": [
            "wsss-train=scripts.train:main",
            "wsss-inference=scripts.inference:main",
            "wsss-visualize=scripts.visualize:main",
        ],
    },
)
