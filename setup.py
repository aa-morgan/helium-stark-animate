from setuptools import setup

setup(name='HeliumStarkAnimate',
        version='0.0.1.dev',
        description='Create animations of the Rydberg electron charge distribution in helium as a function of electric field using the Numerov method.',
        url='',
        author='Alex Morgan',
        author_email='alexandre.morgan.15@ucl.ac.uk',
        license='GPL-3.0',
        packages=['heliumstarkanimate'],
        install_requires=[
            'tqdm',
            'attrs'
        ],
        include_package_data=True,
        zip_safe=False)
