from setuptools import find_packages, setup

package_name = 'plc_worker_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='thinh',
    maintainer_email='quanthinhnguyen2003@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'plc_communicate_node = plc_worker_pkg.plc_communicate_node:main',
            'plc_monitor_node = plc_worker_pkg.plc_monitor_node:main',
            'plc_motion_node = plc_worker_pkg.plc_motion_node:main'
        ],
    },
)
