from setuptools import setup, find_packages

setup(
    name="ml_institute_week_4_image_captioning",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)