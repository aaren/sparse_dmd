try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
        'description': 'Sparse Dynamic Mode Decomposition',
        'author': "Aaron O'Leary",
        'url': 'http://github.com/aaren/sparse_dmd',
        'author_email': 'eeaol@leeds.ac.uk',
        'version': '0.2',
        'packages': ['sparse_dmd'],
        'scripts': [],
        'name': 'sparse_dmd',
        }

setup(**config)
