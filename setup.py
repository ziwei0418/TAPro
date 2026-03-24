"""
Setup configuration for WSSS project
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="wsss-breast-cancer",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Weakly-Supervised Semantic Segmentation for Breast Cancer Histopathology",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/WSSS_2026_RESEARCH",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
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
