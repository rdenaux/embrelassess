from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='embrelassess',
    version='0.1.0',
    description='Assess how well embeddings can predict certain relations',
    long_description=readme,
    author='Ronald Denaux',
    author_email='rdenaux@expertsystem.com',
    url='https://github.com/rdenaux/embrelassess',
    license=license,
    packages=find_packages(exclude=('test', 'doc'))
)
