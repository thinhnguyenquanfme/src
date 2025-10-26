from setuptools import find_packages, setup

package_name = 'camera_worker_pkg'

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
            'basler_snapshot_server = camera_worker_pkg.frame_grab_node:main',
            'basler_calibfile = camera_worker_pkg.camera_calib_node:main',
            'undistort_img_node = camera_worker_pkg.camera_undistort_node:main',
            'canny_edge_node = camera_worker_pkg.canny_edge_node:main',
            'ght_node = camera_worker_pkg.ght_node:main'
        ],
    },
)
