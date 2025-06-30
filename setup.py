"""
📦 Setup Configuration - INE Data Scraper
=========================================

Configuración para instalación como paquete Python.

Author: Bruno San Martín
Date: 2025-06-28
"""

from setuptools import setup, find_packages
from pathlib import Path

# Leer README para descripción larga
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Leer requirements
requirements = []
requirements_path = this_directory / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, 'r', encoding='utf-8') as f:
        requirements = [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith('#') and not line.startswith('-')
        ]

setup(
    name="ine-data-scraper",
    version="1.0.0",
    author="Bruno San Martín",
    author_email="bruno.sanmartin@email.com",
    description="Sistema automatizado de web scraping para datos del INE Chile",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/brunosanmartin/ine-data-scraper",
    project_urls={
        "Bug Tracker": "https://github.com/brunosanmartin/ine-data-scraper/issues",
        "Documentation": "https://github.com/brunosanmartin/ine-data-scraper/wiki",
        "Source Code": "https://github.com/brunosanmartin/ine-data-scraper",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Internet :: WWW/HTTP :: Indexing/Search",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pytest-cov>=4.1.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "plotly>=5.15.0",
        ],
        "all": [
            "pytest>=7.4.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pytest-cov>=4.1.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "plotly>=5.15.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "ine-scraper=scripts.scraping:main",
            "ine-process=src.data_processor:main",
            "ine-analyze=src.data_analyzer:main",
            "ine-report=src.report_generator:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.yaml", "config/*.yml", "config/*.json"],
    },
    keywords=[
        "web-scraping", 
        "data-science", 
        "ine", 
        "chile", 
        "statistics", 
        "selenium", 
        "pandas",
        "data-analysis"
    ],
    zip_safe=False,
)
