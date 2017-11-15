from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='embrelpredict',
    version='0.1.0',
    description='Evaluate how well embeddings can predict certain relations',
    long_description=readme,
    author='Ronald Denaux',
    author_email='rdenaux@gmail.com',
    url='https://gitlab.com/rdenaux/embedding-rel-predict',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
