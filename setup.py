from setuptools import setup, find_packages

# Read dependencies from requirements.txt
def parse_requirements(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

requirements = parse_requirements('requirements.txt')

setup(
    name='vacuumclean',
    version='0.1.0',
    packages=find_packages(),  # auto-discovers your_package/
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'vacuumclean=main:main'
        ]
    },
    author='Hakim Mitiche',
    description='a vacuum cleaner world simulator, with some RL and other agents',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: NO License'
    ],
)