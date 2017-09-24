from distutils.core import setup

setup(
    name='pyserialization',
    description='Human readable serialization for python objects',
    author='Body Vision Medical',
    author_email='engineering@bodyvisionmedical.com',
    url='https://github.com/BodyVisionMedical/pyserialization',
    license='LGPL',
    version='0.1.1',
    test_suite='nose.collector',
    tests_require=['nose'],
    packages=['pyserialization'],
    install_requires=['jsonpickle', 'numpy']
)
