from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'spiderbot_bridge'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
            (os.path.join('share', package_name, 'launch'), 
            glob('launch/*.py')),
            (os.path.join('share', package_name, 'config'), 
            glob('config/*.yaml')),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='vijay',
    maintainer_email='vijay@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
		'pi_bridge_node = spiderbot_bridge.pi_bridge_node:main',
        'pi_hard_node = spiderbot_bridge.pi_hard_node:main',
        ],
    },
)
