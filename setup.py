'''
Created on 04.07.2018

@author: graichen
'''
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

setup(
    name='SpharaPy',
    version='1.0.12',
    description='SPHARA Implementation in Python',
    long_description=readme,
    long_description_content_type='text/x-rst',
    author='Uwe Graichen',
    author_email='uwe.graichen@tu-ilmenau.de',
    url=' ',
    license='BSD-3-Clause',
    packages=find_packages(exclude=('tests', 'docs')),
    package_data={'spharapy': ['datasets/data/*.csv',
                               'datasets/descr/*.rst',
                               'examples/*.py',
                               'examples/*.ipynb']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
    ],

    project_urls={  # Optional
        'Documentation': 'https://spharapy.readthedocs.io/en/latest/',
    },
)
