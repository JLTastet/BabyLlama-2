from setuptools import setup, find_packages

setup(
    name="babyllama2",
    version="0.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    author="Author1, Author2",
    description="BabyLlama2 project for the BabyLM challenge",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    install_requires=[
        # Add your project dependencies here
    ],
    python_requires='>=3.8',
)