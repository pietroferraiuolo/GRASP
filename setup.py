"""
Created on May 2024

    Author: P. Ferraiuolo
"""
import os
import setuptools

about = {}
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'grasp', '__version__.py'), 'r') as _:
    exec(_.read(), about)

with open ('requirements.txt', 'r') as _:
    requires = [line.split()[0] for line in _]
    
setuptools.setup(
    name=about['__title__'],
    version=about['__version__'],
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={
        'grasp': ['sysdata/_Catalogue.xlsx', 'sysdata/base.frm'],
    },
    install_requires=requires,
    author=about['__author__'],
    author_email=about['__author_email__'],
    description=about['__description__'],
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url=about['__url__'],
    license=about['__license__'],
    python_requires='>=3.10',
)
