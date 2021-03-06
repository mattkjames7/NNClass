import setuptools
from setuptools.command.install import install
import os

with open("README.md", "r") as fh:
    long_description = fh.read()

def getversion():
	'''
	read the version string from __init__
	
	'''
	#get the init file path
	thispath = os.path.abspath(os.path.dirname(__file__))+'/'
	initfile = thispath + 'NNClass/__init__.py'
	
	#read the file in
	f = open(initfile,'r')
	lines = f.readlines()
	f.close()
	
	#search for the version
	version = 'unknown'
	for l in lines:
		if '__version__' in l:
			s = l.split('=')
			version = s[-1].strip().strip('"').strip("'")
			break
	return version
	
version = getversion()

setuptools.setup(
    name="NNClass",
    version=version,
    author="Matthew Knight James",
    author_email="mattkjames7@gmail.com",
    description="A simple bit of code for training classification neural networks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mattkjames7/NNClass",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: POSIX",
    ],
    install_requires=[
		'numpy',
		'PyFileIO',
		'matplotlib',
		'tensorflow',
		'scikit-learn',
	],
	include_package_data=True,
)



