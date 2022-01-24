# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


setup(
    name='cleanml',
    version='0.0.1',
    description='CleanML',
    long_description='CleanML',
    author='Not me',
    author_email='neutatz@gmail.com',
    url='https://github.com/chu-data-lab/CleanML',
    license='test',
    include_package_data=True,
    install_requires=[],
    packages=find_packages(exclude=('tests', 'docs'))
)


