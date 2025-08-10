#!/usr/bin/env python3
"""Setup script pour le Système de Surveillance Intelligente."""

from setuptools import setup, find_packages
import os
import sys
from pathlib import Path

# Lecture de la version
def get_version():
    """Récupère la version depuis __init__.py."""
    version_file = Path("src") / "__init__.py"
    if version_file.exists():
        with open(version_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("__version__"):
                    return line.split("=")[1].strip().strip('"\'')
    return "1.0.0"

# Lecture du README
def get_long_description():
    """Récupère la description longue depuis README.md."""
    readme_file = Path("README.md")
    if readme_file.exists():
        with open(readme_file, "r", encoding="utf-8") as f:
            return f.read()
    return "Système de Surveillance Intelligente Multimodale"

# Lecture des requirements
def get_requirements(filename="requirements.txt"):
    """Récupère les dépendances depuis requirements.txt."""
    requirements_file = Path(filename)
    if requirements_file.exists():
        with open(requirements_file, "r", encoding="utf-8") as f:
            return [
                line.strip() for line in f
                if line.strip() and not line.startswith("#")
            ]
    return []

# Vérification Python version
if sys.version_info < (3, 9):
    sys.exit("Python 3.9 ou supérieur requis")

# Configuration
setup(
    name="intelligent-surveillance-system",
    version=get_version(),
    author="Elfried Steve David KINZOUN",
    author_email="elfried.kinzoun@example.com",
    description="Système de surveillance basé sur VLM avec orchestration d'outils",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/elfried-kinzoun/intelligent-surveillance-system",
    project_urls={
        "Documentation": "https://elfried-kinzoun.github.io/intelligent-surveillance-system/",
        "Source Code": "https://github.com/elfried-kinzoun/intelligent-surveillance-system",
        "Issue Tracker": "https://github.com/elfried-kinzoun/intelligent-surveillance-system/issues",
        "Colab Demo": "https://colab.research.google.com/github/elfried-kinzoun/intelligent-surveillance-system/blob/main/notebooks/demo.ipynb",
    },
    
    # Packages et structure
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # Scripts exécutables
    entry_points={
        "console_scripts": [
            "surveillance-system=src.main:main",
            "surveillance-demo=src.main:demo_main",
            "surveillance-benchmark=src.scripts.benchmark:main",
        ],
    },
    
    # Données du package
    package_data={
        "": ["*.yml", "*.yaml", "*.json", "*.toml", "*.txt"],
    },
    include_package_data=True,
    
    # Dépendances
    install_requires=get_requirements("requirements.txt"),
    
    # Dépendances optionnelles
    extras_require={
        "dev": get_requirements("requirements-dev.txt"),
        "docs": get_requirements("requirements-docs.txt"),
        "gpu": [
            "torch[cuda]>=2.1.0",
            "torchvision[cuda]>=0.16.0",
        ],
        "full": get_requirements("requirements.txt") + 
               get_requirements("requirements-dev.txt") +
               get_requirements("requirements-docs.txt"),
        "colab": [
            "torch>=2.1.0",
            "torchvision>=0.16.0",
            "transformers>=4.35.0",
            "ultralytics>=8.0.0",
            "opencv-python-headless>=4.8.0",
            "matplotlib>=3.7.0",
            "ipywidgets>=8.0.0",
        ],
    },
    
    # Configuration Python
    python_requires=">=3.9",
    
    # Métadonnées
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Security",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Typing :: Typed",
    ],
    keywords=[
        "surveillance", "ai", "computer-vision", "yolo", "tracking",
        "vlm", "vision-language", "tool-calling", "security", "retail",
        "false-positive-reduction", "real-time", "pytorch", "transformers"
    ],
    
    # Options ZIP
    zip_safe=False,
    
    # Tests
    test_suite="tests",
    
    # Configuration pour PyPI
    platforms=["any"],
)

# Post-installation hooks
def post_install():
    """Actions post-installation."""
    print("🎉 Installation terminée!")
    print("📖 Documentation: https://elfried-kinzoun.github.io/intelligent-surveillance-system/")
    print("🧪 Test Colab: https://colab.research.google.com/github/elfried-kinzoun/intelligent-surveillance-system/blob/main/notebooks/demo.ipynb")
    print("🚀 Commande de démo: surveillance-demo")

if __name__ == "__main__":
    post_install()