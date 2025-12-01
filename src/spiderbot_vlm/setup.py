from setuptools import find_packages, setup

package_name = 'spiderbot_vlm'

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
    maintainer='teja',
    maintainer_email='btvvardhan@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vlm_scene_planner_node = spiderbot_vlm.vlm_scene_planner_node:main',
        ],
    },
)
