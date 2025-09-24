from setuptools import setup, find_packages

setup(
    name="cfwt_ccews",
    version="0.1.0",
    author="Chang Xu",
    author_email="chang.xu8@unsw.edu.au",
    description="Combined Fourier-Wavelet Transform tools for Convectively Coupled Waves",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Changbaober/pyCFWT",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "xarray",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
