from setuptools import setup, find_packages

setup(
    name="ml_institute_week_6_fine_tuning",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)