#pragma once

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <gum/graph/types.hpp>
#include <gum/perception/bbox/bbox.h>
#include <gum/perception/dataset/dataset.h>
#include <gum/perception/feature/frame.cuh>
#include <gum/perception/feature/light_glue.h>
#include <gum/perception/feature/outlier_rejection.h>
#include <gum/perception/feature/super_point.h>
#include <gum/perception/segmentation/segmentation.h>

namespace gum {
namespace perception {
class SAMPublisher : public rclcpp::Node {
public:
  using Frame = gum::perception::feature::Frame;
  SAMPublisher(const std::string &node_name);

  Frame &Initialize(const cv::Mat &image, const cv::Mat &depth,
                    const cv::Mat &mask);
  void Process(const Frame &prev_frame, Frame &curr_frame);

protected:
  using ApproximatePolicy = message_filters::sync_policies::ApproximateTime<
      sensor_msgs::msg::CompressedImage, sensor_msgs::msg::Image>;

  std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::Image>>
      m_segmentation_publisher;
  std::shared_ptr<
      message_filters::Subscriber<sensor_msgs::msg::CompressedImage>>
      m_color_subscriber;
  std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>>
      m_depth_subscriber;
  std::shared_ptr<message_filters::Synchronizer<ApproximatePolicy>>
      m_synchronizer;

  std::shared_ptr<gum::perception::segmentation::SAM> m_sam;
  std::shared_ptr<gum::perception::feature::SuperPoint> m_superpoint;
  std::shared_ptr<gum::perception::feature::LightGlue> m_lightglue;
  std::shared_ptr<gum::perception::bbox::OSTrack> m_ostracker;
  std::shared_ptr<gum::perception::dataset::RealSenseDataset<
      gum::perception::dataset::Device::GPU>>
      m_dataset;

  int m_device;
  int m_height, m_width;
  Eigen::Vector<float, 8> m_intrinsics;
  float m_depth_scale, m_min_depth, m_max_depth;
  std::shared_ptr<gum::graph::Handle> m_handle;
  gum::perception::feature::GraphParameters m_graph_params;
  gum::perception::feature::LeidenParameters m_leiden_params;
  float m_outlier_tolerance;

  void CallBack(const sensor_msgs::msg::CompressedImage::ConstSharedPtr &color,
                const sensor_msgs::msg::Image::ConstSharedPtr &depth);
};
} // namespace perception
} // namespace gum