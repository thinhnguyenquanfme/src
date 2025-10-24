from setuptools import find_packages, setup

package_name = 'test_node_pkg'

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
    maintainer_email='thinh@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "publisher_node = test_node_pkg.publisher_node:main",
            "subscriber_node = test_node_pkg.subscriber_node:main",
            "turtlesim_control_node = test_node_pkg.turtlesim_control_node:main",
            "service_control_node = test_node_pkg.service_node:main"
            
        ],
    },
)
