#include "livox_ros_driver2/msg/custom_msg.hpp"
// 删除手动buffer赋值，并判断topic在反序列化，同时增加空消息、非法点检查，同时支持CustomMsg和pc2
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/serialization.hpp>
#include <rosbag2_cpp/converter_options.hpp>
#include <rosbag2_cpp/readers/sequential_reader.hpp>
#include <rosbag2_storage/storage_options.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

namespace {

bool isFinitePoint(float x, float y, float z) {
  return std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
}

}  // namespace

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("bag_to_pcd");

  node->declare_parameter<std::string>("bag_file", "");
  node->declare_parameter<std::string>("pcd_file", "");
  node->declare_parameter<std::string>("lidar_topic", "/livox/lidar");
  node->declare_parameter<bool>("is_custom_msg", false);

  const std::string bag_file = node->get_parameter("bag_file").as_string();
  const std::string pcd_file = node->get_parameter("pcd_file").as_string();
  const std::string lidar_topic = node->get_parameter("lidar_topic").as_string();
  const bool is_custom_msg = node->get_parameter("is_custom_msg").as_bool();

  if (bag_file.empty()) {
    RCLCPP_ERROR(node->get_logger(), "Parameter 'bag_file' is empty");
    rclcpp::shutdown();
    return 1;
  }

  if (pcd_file.empty()) {
    RCLCPP_ERROR(node->get_logger(), "Parameter 'pcd_file' is empty");
    rclcpp::shutdown();
    return 1;
  }

  if (lidar_topic.empty()) {
    RCLCPP_ERROR(node->get_logger(), "Parameter 'lidar_topic' is empty");
    rclcpp::shutdown();
    return 1;
  }

  RCLCPP_INFO(node->get_logger(), "Loading rosbag: %s", bag_file.c_str());
  RCLCPP_INFO(node->get_logger(), "Lidar topic: %s", lidar_topic.c_str());
  RCLCPP_INFO(node->get_logger(), "Output PCD: %s", pcd_file.c_str());
  RCLCPP_INFO(node->get_logger(), "is_custom_msg: %s", is_custom_msg ? "true" : "false");

  rosbag2_cpp::readers::SequentialReader reader;

  rosbag2_storage::StorageOptions storage_options;
  storage_options.uri = bag_file;
  storage_options.storage_id = "sqlite3";

  rosbag2_cpp::ConverterOptions converter_options;
  converter_options.input_serialization_format = "cdr";
  converter_options.output_serialization_format = "cdr";

  try {
    reader.open(storage_options, converter_options);
  } catch (const std::exception & e) {
    RCLCPP_ERROR(node->get_logger(), "Failed to open rosbag '%s': %s", bag_file.c_str(), e.what());
    rclcpp::shutdown();
    return 1;
  }

  pcl::PointCloud<pcl::PointXYZI> output_cloud;
  output_cloud.is_dense = false;
  output_cloud.height = 1;

  size_t lidar_msg_count = 0;
  size_t deserialization_failed_count = 0;
  size_t skipped_invalid_point_count = 0;

  rclcpp::Serialization<livox_ros_driver2::msg::CustomMsg> custom_serializer;
  rclcpp::Serialization<sensor_msgs::msg::PointCloud2> pointcloud2_serializer;

  while (reader.has_next() && rclcpp::ok()) {
    auto bag_message = reader.read_next();

    if (!bag_message) {
      continue;
    }

    // Important: a bag may contain camera/image/tf topics. Do not deserialize
    // other topics as lidar messages.
    if (bag_message->topic_name != lidar_topic) {
      continue;
    }

    ++lidar_msg_count;

    if (!bag_message->serialized_data || bag_message->serialized_data->buffer_length == 0) {
      RCLCPP_WARN(node->get_logger(), "Skip empty serialized message on topic %s", lidar_topic.c_str());
      continue;
    }

    if (is_custom_msg) {
      livox_ros_driver2::msg::CustomMsg livox_msg;

      try {
        // Do NOT manually assign serialized_msg.get_rcl_serialized_message().buffer.
        // That can make rclcpp free memory owned by rosbag2 and cause exit code -6.
        rclcpp::SerializedMessage serialized_msg(*bag_message->serialized_data);
        custom_serializer.deserialize_message(&serialized_msg, &livox_msg);
      } catch (const std::exception & e) {
        ++deserialization_failed_count;
        RCLCPP_ERROR(
          node->get_logger(),
          "Failed to deserialize %s as livox_ros_driver2/msg/CustomMsg: %s",
          lidar_topic.c_str(), e.what());
        continue;
      }

      const size_t point_num = static_cast<size_t>(livox_msg.point_num);
      const size_t points_size = livox_msg.points.size();
      const size_t n = std::min(point_num, points_size);

      if (point_num != points_size) {
        RCLCPP_WARN(
          node->get_logger(),
          "CustomMsg point_num=%zu but points.size()=%zu, using %zu",
          point_num, points_size, n);
      }

      for (size_t i = 0; i < n; ++i) {
        const auto & src = livox_msg.points[i];

        if (!isFinitePoint(src.x, src.y, src.z)) {
          ++skipped_invalid_point_count;
          continue;
        }

        pcl::PointXYZI p;
        p.x = src.x;
        p.y = src.y;
        p.z = src.z;
        p.intensity = static_cast<float>(src.reflectivity);
        output_cloud.points.push_back(p);
      }
    } else {
      sensor_msgs::msg::PointCloud2 cloud_msg;

      try {
        // Safe copy from rosbag2 serialized buffer.
        rclcpp::SerializedMessage serialized_msg(*bag_message->serialized_data);
        pointcloud2_serializer.deserialize_message(&serialized_msg, &cloud_msg);
      } catch (const std::exception & e) {
        ++deserialization_failed_count;
        RCLCPP_ERROR(
          node->get_logger(),
          "Failed to deserialize %s as sensor_msgs/msg/PointCloud2: %s",
          lidar_topic.c_str(), e.what());
        continue;
      }

      pcl::PCLPointCloud2 pcl_pc2;
      pcl_conversions::toPCL(cloud_msg, pcl_pc2);

      pcl::PointCloud<pcl::PointXYZI> frame_cloud;
      try {
        pcl::fromPCLPointCloud2(pcl_pc2, frame_cloud);
      } catch (const std::exception & e) {
        RCLCPP_ERROR(node->get_logger(), "Failed to convert PointCloud2 to pcl::PointXYZI: %s", e.what());
        continue;
      }

      for (const auto & p : frame_cloud.points) {
        if (!isFinitePoint(p.x, p.y, p.z)) {
          ++skipped_invalid_point_count;
          continue;
        }
        output_cloud.points.push_back(p);
      }
    }
  }

  output_cloud.width = static_cast<uint32_t>(output_cloud.points.size());
  output_cloud.height = 1;
  output_cloud.is_dense = false;

  RCLCPP_INFO(node->get_logger(), "Matched lidar messages: %zu", lidar_msg_count);
  RCLCPP_INFO(node->get_logger(), "Total valid points: %zu", output_cloud.points.size());

  if (deserialization_failed_count > 0) {
    RCLCPP_WARN(node->get_logger(), "Deserialization failed messages: %zu", deserialization_failed_count);
  }

  if (skipped_invalid_point_count > 0) {
    RCLCPP_WARN(node->get_logger(), "Skipped invalid points: %zu", skipped_invalid_point_count);
  }

  if (lidar_msg_count == 0) {
    RCLCPP_ERROR(node->get_logger(), "No messages found for lidar_topic: %s", lidar_topic.c_str());
    rclcpp::shutdown();
    return 1;
  }

  if (output_cloud.points.empty()) {
    RCLCPP_ERROR(node->get_logger(), "No valid points found in topic: %s", lidar_topic.c_str());
    rclcpp::shutdown();
    return 1;
  }

  const int ret = pcl::io::savePCDFileASCII(pcd_file, output_cloud);
  if (ret != 0) {
    RCLCPP_ERROR(node->get_logger(), "Failed to save PCD file: %s", pcd_file.c_str());
    rclcpp::shutdown();
    return 1;
  }

  RCLCPP_INFO(node->get_logger(), "Successfully saved point cloud to PCD file: %s", pcd_file.c_str());
  RCLCPP_INFO(node->get_logger(), "Total points saved: %zu", output_cloud.points.size());

  rclcpp::shutdown();
  return 0;
}