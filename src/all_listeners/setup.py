## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# fetch values from package.xml
setup_args = generate_distutils_setup(
    packages=['all_listeners'],
    package_dir={'': 'src'},
    requires=['std_msgs', 'rospy', 'cv_bridge', 'image_transport', 'sensor_msgs']
)

setup(**setup_args)