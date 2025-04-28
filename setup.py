from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="deepview",
    version="0.0.1",
    description="Spyre AIU models debugging tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/IBM/deepview.git", 
    packages=find_packages(),
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'deepview=deepview:main',  
        ],
    },
    install_requires=[],
)