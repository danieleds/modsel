from setuptools import setup

setup(name='modsel',
      version='0.1.3',
      description='A simple model selection tool for machine learning experiments',
      url='http://github.com/danieleds/modsel',
      author='Daniele Di Sarli',
      author_email='danieleds0@gmail.com',
      license='MIT',
      packages=['modsel'],
      entry_points={
            'console_scripts': ['modsel=modsel.command_line:main'],
      },
      install_requires=[
            'ruamel.yaml',
            'numpy',
            'hyperopt',
            'ray[tune]'
      ],
      zip_safe=True)
