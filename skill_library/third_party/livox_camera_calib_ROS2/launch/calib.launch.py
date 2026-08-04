from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Get package directory
    pkg_share = FindPackageShare('livox_camera_calib')
    
    # Declare launch arguments
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=PathJoinSubstitution([pkg_share, 'config', 'calib.yaml']),
        description='Path to the calibration config file'
    )
    
    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config',
        default_value=PathJoinSubstitution([pkg_share, 'rviz_cfg', 'calib.rviz']),
        description='Path to the RViz config file'
    )
    
    # Calibration node
    calib_node = Node(
        package='livox_camera_calib',
        executable='lidar_camera_calib',
        name='lidar_camera_calib',
        output='screen',
        parameters=[LaunchConfiguration('config_file')]
    )
    
    # RViz node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', LaunchConfiguration('rviz_config')],
        output='screen'
    )
    
    return LaunchDescription([
        config_file_arg,
        rviz_config_arg,
        calib_node,
        rviz_node
    ])