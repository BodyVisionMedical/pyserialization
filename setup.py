from setuptools import setup

setup(
    name='pyserialization',
    description='Human readable serialization for python objects',
    author='Body Vision Medical',
    author_email='engineering@bodyvisionmedical.com',
    url='https://github.com/BodyVisionMedical/pyserialization',
    license='LGPL',
    version='0.1.2',
    test_suite='nose.collector',
    tests_require=['nose', 'testfixtures],
    packages=['pyserialization'],
    install_requires=['jsonpickle', 'numpy'],
    python_requires=">=3.5",
)
