#pragma once

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/char.hpp>

#include <pinocchio/algorithm/frames.hpp>

#include <gum/graph/types.hpp>
#include <gum/perception/bbox/bbox.h>
#include <gum/perception/camera/camera.h>
#include <gum/perception/feature/fast_super_point.h>
#include <gum/perception/feature/frame.cuh>
#include <gum/perception/feature/light_glue.h>
#include <gum/perception/feature/outlier_rejection.h>
#include <gum/perception/segmentation/segmentation.h>

namespace gum {
namespace perception {
template <typename ColorMsg, typename DepthMsg>
class SAMPublisher : public rclcpp::Node {
public:
  using Frame = gum::perception::feature::Frame;
  using FramePtr = std::shared_ptr<Frame>;
  using FrameConstPtr = std::shared_ptr<const Frame>;
  using JointMsg = sensor_msgs::msg::JointState;
  using ImageMsg = sensor_msgs::msg::Image;

  SAMPublisher(const std::string &node_name);

  void Reset() const;

protected:
  using ApproximatePolicy =
      message_filters::sync_policies::ApproximateTime<ColorMsg, DepthMsg,
                                                      JointMsg>;
  using Synchronizer = message_filters::Synchronizer<ApproximatePolicy>;

  std::shared_ptr<rclcpp::Publisher<ImageMsg>> m_seg_depth_publisher;
  std::shared_ptr<message_filters::Subscriber<ColorMsg>> m_color_subscriber;
  std::shared_ptr<message_filters::Subscriber<DepthMsg>> m_depth_subscriber;
  std::shared_ptr<message_filters::Subscriber<JointMsg>> m_joint_subscriber;
  std::shared_ptr<Synchronizer> m_synchronizer;
  std::shared_ptr<rclcpp::Subscription<std_msgs::msg::Char>> m_cmd_subscriber;

  std::shared_ptr<gum::perception::segmentation::SAM> m_sam;
  std::shared_ptr<gum::perception::segmentation::SAM> m_mobile_sam;
  std::shared_ptr<gum::perception::feature::FastSuperPoint> m_superpoint;
  std::shared_ptr<gum::perception::feature::LightGlue> m_lightglue;
  std::shared_ptr<gum::perception::bbox::OSTrack> m_ostracker;

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
  double m_sam_offset;

  mutable std::vector<Eigen::Vector2f> m_initial_keypoints_v;
  mutable std::vector<float> m_initial_keypoint_scores_v;
  mutable std::vector<Eigen::Vector<float, 256>> m_initial_descriptors_v;
  mutable std::vector<Eigen::Vector2f> m_initial_normalized_keypoints_v;
  mutable std::vector<Eigen::Vector3f> m_initial_point_clouds_v;

  mutable std::shared_ptr<gum::perception::camera::RealSenseCamera<
      gum::perception::camera::Device::GPU>>
      m_realsense_camera;
  mutable std::vector<Eigen::VectorXd> m_joint_angles_v;
  mutable std::vector<FramePtr> m_frames_v;
  mutable int m_num_frames = 0;

  mutable bool m_started = false;
  mutable bool m_active = false;

  int m_save_results = 0;

private:
  void Clear() const;

  void Initialize(const cv::Mat &image, const cv::Mat &depth,
                  const Eigen::VectorXd &joint_angles,
                  FramePtr curr_frame) const;
  void Iterate(const cv::Mat &image, const cv::Mat &depth,
               const Eigen::VectorXd &joint_angles, FrameConstPtr prev_frame,
               FramePtr curr_frame) const;
  void WarmUp() const;

  void AddFrame(typename ColorMsg::ConstSharedPtr color_msg,
                typename DepthMsg::ConstSharedPtr depth_msg,
                typename JointMsg::ConstSharedPtr joint_msg) const;
  void ProjectGraspCenter(const std::vector<Eigen::Vector3d> &finger_tips,
                          std::vector<Eigen::Vector2d> &finger_tip_centers,
                          Eigen::Vector2d &grasp_center) const;
  void GetFingerTips(const Eigen::VectorXd &joint_angles,
                     std::vector<Eigen::Vector3d> &finger_tips) const;
  void ExtractKeyPoints(FramePtr frame, const uint8_t *mask_ptr) const;
  void RefineKeyPoints(FramePtr frame) const;
  void WriteFrame(FrameConstPtr frame) const;
  void WriteMatch(FrameConstPtr prev_frame, FrameConstPtr curr_frame,
                  const std::vector<Eigen::Vector2i> &matches_v) const;
  void Publish(FrameConstPtr frame, const std_msgs::msg::Header &header) const;
  void SensorCallBack(typename ColorMsg::ConstSharedPtr color_msg,
                      typename DepthMsg::ConstSharedPtr depth_msg,
                      typename JointMsg::ConstSharedPtr joint_msg) const;
  void CommandCallBack(std_msgs::msg::Char cmd_msg) const;
};
} // namespace perception
} // namespace gum