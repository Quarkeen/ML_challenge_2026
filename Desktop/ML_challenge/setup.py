from setuptools import setup, find_packages

setup(
    name="entity_resolution",
    version="1.0.0",
    description="High-Precision Multi-Source Entity Resolution System optimizing Macro F_0.5",
    author="Entity Resolution Implementation Lead",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.2.0",
        "rapidfuzz>=3.0.0",
        "xgboost>=2.0.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "gpu": ["lightgbm>=4.0.0"],
        "dev": ["pytest>=7.0.0"],
    },
    entry_points={
        "console_scripts": [
            "er-train=entity_resolution.train:run_training_pipeline",
            "er-predict=entity_resolution.predict:generate_predictions",
            "er-baselines=entity_resolution.baselines:run_baselines",
        ],
    },
)
