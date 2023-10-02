#pragma once

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

#include <pinocchio/algorithm/frames.hpp>

#include <gum/graph/types.hpp>
#include <gum/perception/bbox/bbox.h>
#include <gum/perception/dataset/dataset.h>
#include <gum/perception/feature/frame.cuh>
#include <gum/perception/feature/light_glue.h>
#include <gum/perception/feature/outlier_rejection.h>
#include <gum/perception/feature/fast_super_point.h>
#include <gum/perception/segmentation/segmentation.h>

namespace gum {
namespace perception {
class SAMPublisher : public rclcpp::Node {
public:
  using Frame = gum::perception::feature::Frame;
  SAMPublisher(const std::string &node_name);

protected:
  using ApproximatePolicy = message_filters::sync_policies::ApproximateTime<
      sensor_msgs::msg::CompressedImage, sensor_msgs::msg::Image,
      sensor_msgs::msg::JointState>;
  using Synchronizer = message_filters::Synchronizer<ApproximatePolicy>;

  std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::Image>>
      m_segmentation_publisher;
  std::shared_ptr<
      message_filters::Subscriber<sensor_msgs::msg::CompressedImage>>
      m_color_subscriber;
  std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>>
      m_depth_subscriber;
  std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::JointState>>
      m_joint_subscriber;
  std::shared_ptr<Synchronizer> m_synchronizer;

  std::shared_ptr<gum::perception::segmentation::SAM> m_sam;
  std::shared_ptr<gum::perception::segmentation::SAM> m_mobile_sam;
  std::shared_ptr<gum::perception::feature::FastSuperPoint> m_superpoint;
  std::shared_ptr<gum::perception::feature::LightGlue> m_lightglue;
  std::shared_ptr<gum::perception::bbox::OSTrack> m_ostracker;

  std::shared_ptr<gum::perception::dataset::RealSenseDataset<
      gum::perception::dataset::Device::GPU>>
      m_realsense;
  std::vector<Eigen::VectorXd> m_joint_angles_v;

  int m_device;
  int m_height, m_width;
  Eigen::Vector<float, 8> m_intrinsics;
  float m_depth_scale, m_min_depth, m_max_depth;
  std::shared_ptr<gum::graph::Handle> m_handle;
  gum::perception::feature::GraphParameters m_graph_params;
  gum::perception::feature::LeidenParameters m_leiden_params;
  float m_outlier_tolerance;
  std::string m_result_path;

  pinocchio::Model m_robot_model;
  Eigen::Matrix<double, 3, 4> m_base_pose;
  Eigen::Vector3d m_finger_offset;
  std::vector<int> m_finger_ids;
  Eigen::Matrix<double, 3, 4> m_pose_wc;

  std::vector<Eigen::Vector2f> m_initial_keypoints_v;
  std::vector<float> m_initial_keypoint_scores_v;
  std::vector<Eigen::Vector<float, 256>> m_initial_descriptors_v;
  std::vector<Eigen::Vector2f> m_initial_normalized_keypoints_v;
  std::vector<Eigen::Vector3f> m_initial_point_clouds_v;

  std::vector<Frame> m_frames_v;
  bool m_save_results;

private:
  void Initialize(const cv::Mat &image, const cv::Mat &depth,
                  const Eigen::VectorXd &joint_angles);
  void Process(const cv::Mat &image, const cv::Mat &depth,
               const Eigen::VectorXd &joint_angles);
  void WarmUp();

  void
  AddFrame(const sensor_msgs::msg::CompressedImage::ConstSharedPtr &color_msg,
           const sensor_msgs::msg::Image::ConstSharedPtr &depth_msg,
           const sensor_msgs::msg::JointState::ConstSharedPtr &joint_msg);
  void ProjectGraspCenter(const std::vector<Eigen::Vector3d> &finger_tips,
                          Eigen::Vector2d &grasp_center);
  void GetFingerTips(const Eigen::VectorXd &joint_angles,
                     std::vector<Eigen::Vector3d> &finger_tips);
  void ExtractKeyPoints(Frame &frame, const uint8_t *mask_ptr);
  void RefineKeyPoints(Frame &frame);
  void WriteFrame(const Frame &frame);
  void Publish(const Frame &frame, const std_msgs::msg::Header &header);
  void
  CallBack(const sensor_msgs::msg::CompressedImage::ConstSharedPtr &color_msg,
           const sensor_msgs::msg::Image::ConstSharedPtr &depth_msg,
           const sensor_msgs::msg::JointState::ConstSharedPtr &joint_msg);
};
} // namespace perception
} // namespace gum