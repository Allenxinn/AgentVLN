from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Get package directory
    pkg_share = FindPackageShare('livox_camera_calib')
    
    # Declare launch arguments
    bag_file_arg = DeclareLaunchArgument(
        'bag_file',
        default_value='/home/ycj/data/calib/outdoor1.bag',
        description='Path to the bag file'
    )
    
    lidar_topic_arg = DeclareLaunchArgument(
        'lidar_topic',
        default_value='/livox/lidar',
        description='Lidar topic name'
    )
    
    pcd_file_arg = DeclareLaunchArgument(
        'pcd_file',
        default_value=PathJoinSubstitution([pkg_share, 'result', '0.pcd']),
        description='Output PCD file path'
    )
    
    is_custom_msg_arg = DeclareLaunchArgument(
        'is_custom_msg',
        default_value='false',
        description='Whether to use custom message format'
    )
    
    # bag_to_pcd node
    bag_to_pcd_node = Node(
        package='livox_camera_calib',
        executable='bag_to_pcd',
        name='bag_to_pcd',
        output='screen',
        parameters=[
            {'bag_file': LaunchConfiguration('bag_file')},
            {'lidar_topic': LaunchConfiguration('lidar_topic')},
            {'pcd_file': LaunchConfiguration('pcd_file')},
            {'is_custom_msg': LaunchConfiguration('is_custom_msg')}
        ]
    )
    
    return LaunchDescription([
        bag_file_arg,
        lidar_topic_arg,
        pcd_file_arg,
        is_custom_msg_arg,
        bag_to_pcd_node
    ])