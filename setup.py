from setuptools import setup
import setuptools
from itertools import chain
import os
import sys

with open("requirements.txt", "r") as f:
    requirements = f.readlines()


def find_packages():
    packages = setuptools.find_packages()
    return packages


setup(
    name='selfdrive',
    version='0.1',
    description='self-driving models compatible with CARLA simulator',
    url='#',
    author='Ori David',
    author_email='orid2004@gmail.com',
    license='MIT',
    packages=find_packages(),
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    include_package_data=True,
    install_requires=requirements,
    zip_safe=False
)
