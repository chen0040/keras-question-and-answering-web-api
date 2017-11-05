from setuptools import setup

setup(
    name='qa_system_web',
    packages=['qa_system_web'],
    include_package_data=True,
    install_requires=[
        'flask',
        'keras',
        'sklearn'
    ],
    setup_requires=[
        'pytest-runner',
    ],
    tests_require=[
        'pytest',
    ],
)