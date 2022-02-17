from setuptools import setup, find_packages

with open("selfdrive/requirements.txt", "r") as f:
    requirements = f.readlines()

setup(
    name='selfdrive',
    version='0.1',
    description='self-driving models compatible with CARLA simulator',
    url='#',
    author='Ori David',
    author_email='orid2004@gmail.com',
    license='MIT',
    packages=find_packages(),
    install_requires=requirements,
    zip_safe=False
)