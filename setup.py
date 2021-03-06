#!/usr/bin/env python3

from setuptools import setup


"""
Setup RNAProt

NOTE that additional libraries are needed to run RNAProt. For full 
installation instructions see the README.md at:
https://github.com/BackofenLab/RNAProt

"""

setup(
    name='rnaprot',
    version='0.5',
    description='Modelling RBP binding preferences to predict RPB binding sites',
    long_description=open('README.md').read(),
    author='Michael Uhl',
    author_email='uhlm@informatik.uni-freiburg.de',
    url='https://github.com/BackofenLab/RNAProt',
    license='MIT',
    scripts=['bin/rnaprot', 'bin/gtf_extract_gene_regions.py', 'bin/gtf_extract_transcript_regions.py'],
    packages=['rnaprot'],
    package_data={'rnaprot': ['content/*']},
    zip_safe=False,
)

