from setuptools import setup

setup(
    name='alain-utils-package',
    version='0.1.0',
    description='Python Alain Utils Package',
    author='Alain Schöbi',
    author_email='alain.schoebi@gmx.ch',
    packages=['alutils'],
    install_requires=[
        'numpy',
        'colorama',
        'scipy'
    ],
)