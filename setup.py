from setuptools import setup, find_packages

setup(
    name='jmpy',
    version='1.0.2',
    url='https://github.com/jmoudrik/jmpy.git',
    author='Josef Moudřík',
    author_email='j.moudrik@gmail.com',
    description="jm's Python utils",
    packages=['jmpy'],
    install_requires=['numpy >= 1.11.1'],
)
