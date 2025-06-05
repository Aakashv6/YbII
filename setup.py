from setuptools import setup, find_packages

setup(
    name='YbII',           # Name of your library
    version='1.0.0',                      # Version
    package_dir={'': 'src'},  # Specify the source directory
    packages=find_packages(where='src'),            # Automatically include all packages
    install_requires=[],                 # Dependencies, if any
    author='YbII',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Adjust license as needed
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)