from setuptools import find_packages, setup

package_name = 'system_sim_pkg'

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
            'object_spawn_node = system_sim_pkg.object_spawn_node:main',
            'robot_state_sim_node = system_sim_pkg.robot_state_sim_node:main',
            'point_sim = system_sim_pkg.point_sim:main',
            'trajectory_plotter = system_sim_pkg.trajectory_plotter:main'
        ],
    },
)
