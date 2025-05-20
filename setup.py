from setuptools import setup, find_packages

setup(
    name="ml_institute_week_6_fine_tuning",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "peft>=0.15.2",
        "trl>=0.17.0",
        "torch>=2.0.0",
        "transformers>=4.51.3",
        "datasets>=3.6.0",
    ],
)