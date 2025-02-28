"""Installer for geodesic interpolation package.
Install the package into python environment, and provide an entry point for the
main interpolation script.
"""
from setuptools import setup
import pathlib

# Read the contents of README.md
here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")
 
setup(
    name='geodesic_interpolate',
    version='1.0.0',
    description='Interpolation and smoothing of reaction paths with geodesics in redundant internal coordinates.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Xiaolei Zhu',
    author_email='virtualzx@gmail.com',
    url='https://github.com/virtualzx-nad/geodesic-interpolate',
    license='MIT',
    packages=['geodesic_interpolate'],
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.13',
        'scipy>=0.19',
    ],
    entry_points = {
        'console_scripts': [
            'geodesic_interpolate=geodesic_interpolate.__main__:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Physics',
    ],
    keywords='chemistry, molecular dynamics, reaction paths, geodesics',
    project_urls={
        'Bug Reports': 'https://github.com/virtualzx-nad/geodesic-interpolate/issues',
        'Source': 'https://github.com/virtualzx-nad/geodesic-interpolate',
    },
)
