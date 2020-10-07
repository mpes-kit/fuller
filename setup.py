from setuptools import setup, find_packages
from codecs import open
from os import path

__version__ = '0.9.9'

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md')) as f:
    long_description = f.read()

# get the dependencies and installs
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

install_requires = [x.strip() for x in all_reqs if 'git+' not in x]
dependency_links = [x.strip().replace('git+', '') for x in all_reqs if x.startswith('git+')]

setup(
    name='fuller',
    version=__version__,
    description='Integrated computational framework for electronic band structure reconstruction and parametrization',
    long_description_content_type='text/markdown',
    long_description=long_description,
    url='https://github.com/mpes-kit/fuller',
    download_url='https://github.com/mpes-kit/fuller/tarball/' + __version__,
    license='MIT',
    classifiers=[
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Developers',
      'Programming Language :: Python :: 3',
    ],
    keywords='',
    packages=find_packages(exclude=['docs', 'tests*']),
    include_package_data=True,
    author=['Vincent Stimper', 'R. Patrick Xian'],
    install_requires=install_requires,
    dependency_links=dependency_links,
    author_email=''
)
