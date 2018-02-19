from setuptools import setup

setup(
    name='keras_question_and_answering_system',
    packages=['keras_question_and_answering_system'],
    include_package_data=True,
    install_requires=[
        'flask',
        'keras',
        'scikit-learn'
    ],
    setup_requires=[
        'pytest-runner',
    ],
    tests_require=[
        'pytest',
    ],
)