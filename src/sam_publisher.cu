#include <gum_perception/sam_publisher.h>

#include <cv_bridge/cv_bridge.h>
#include <gum/perception/feature/outlier_rejection.h>
#include <gum/perception/utils/utils.cuh>
#include <gum/utils/cuda_utils.cuh>
#include <gum/utils/kinematics.h>
#include <gum/utils/utils.h>
#include <opencv2/imgproc.hpp>
#include <pinocchio/parsers/urdf.hpp>

namespace gum {
namespace perception {
SAMPublisher::SAMPublisher(const std::string &node_name)
    : rclcpp::Node(node_name) {
  // Declare Parameters
  this->declare_parameter("device", rclcpp::PARAMETER_INTEGER);
  this->declare_parameter("height", rclcpp::PARAMETER_INTEGER);
  this->declare_parameter("width", rclcpp::PARAMETER_INTEGER);
  this->declare_parameter("intrinsics", rclcpp::PARAMETER_DOUBLE_ARRAY);
  this->declare_parameter("depth_scale", rclcpp::PARAMETER_DOUBLE);
  this->declare_parameter("min_depth", rclcpp::PARAMETER_DOUBLE);
  this->declare_parameter("max_depth", rclcpp::PARAMETER_DOUBLE);
  this->declare_parameter("match_graph_delta", rclcpp::PARAMETER_DOUBLE);
  this->declare_parameter("match_graph_tolerance", rclcpp::PARAMETER_DOUBLE);
  this->declare_parameter("leiden_max_iters", rclcpp::PARAMETER_INTEGER);
  this->declare_parameter("leiden_beta", rclcpp::PARAMETER_DOUBLE);
  this->declare_parameter("leiden_resolution", rclcpp::PARAMETER_DOUBLE);
  this->declare_parameter("outlier_tolerance", rclcpp::PARAMETER_DOUBLE);
  this->declare_parameter("base_pose", rclcpp::PARAMETER_DOUBLE_ARRAY);
  this->declare_parameter("pose_wc", rclcpp::PARAMETER_DOUBLE_ARRAY);
  this->declare_parameter("finger_offset", rclcpp::PARAMETER_DOUBLE_ARRAY);
  this->declare_parameter("finger_ids", rclcpp::PARAMETER_INTEGER_ARRAY);

  this->declare_parameter("color_topic", rclcpp::PARAMETER_STRING);
  this->declare_parameter("depth_topic", rclcpp::PARAMETER_STRING);
  this->declare_parameter("joint_state_topic", rclcpp::PARAMETER_STRING);
  this->declare_parameter("segmentation_topic", rclcpp::PARAMETER_STRING);
  this->declare_parameter("meta_hand_urdf", rclcpp::PARAMETER_STRING);
  this->declare_parameter("model_path", rclcpp::PARAMETER_STRING);
  this->declare_parameter("sam_encoder", rclcpp::PARAMETER_STRING);
  this->declare_parameter("sam_decoder", rclcpp::PARAMETER_STRING);
  this->declare_parameter("mobile_sam_encoder", rclcpp::PARAMETER_STRING);
  this->declare_parameter("mobile_sam_decoder", rclcpp::PARAMETER_STRING);
  this->declare_parameter("superpoint", rclcpp::PARAMETER_STRING);
  this->declare_parameter("lightglue", rclcpp::PARAMETER_STRING);
  this->declare_parameter("ostrack", rclcpp::PARAMETER_STRING);
  this->declare_parameter("trt_engine_cache", rclcpp::PARAMETER_STRING);
  this->declare_parameter("test_image", rclcpp::PARAMETER_STRING);

  // Get Parameters
  m_device = this->get_parameter("device").as_int();
  m_height = this->get_parameter("height").as_int();
  m_width = this->get_parameter("width").as_int();
  m_intrinsics = Eigen::Map<const Eigen::Vector<double, 8>>(
                     this->get_parameter("intrinsics").as_double_array().data())
                     .cast<float>();
  m_depth_scale = this->get_parameter("depth_scale").as_double();
  m_min_depth = this->get_parameter("min_depth").as_double();
  m_max_depth = this->get_parameter("max_depth").as_double();
  m_graph_params.delta = this->get_parameter("match_graph_delta").as_double();
  m_graph_params.tolerance =
      this->get_parameter("match_graph_tolerance").as_double();
  m_leiden_params.max_iters = this->get_parameter("leiden_max_iters").as_int();
  m_leiden_params.beta = this->get_parameter("leiden_beta").as_double();
  m_leiden_params.resolution =
      this->get_parameter("leiden_resolution").as_double();
  m_outlier_tolerance = this->get_parameter("outlier_tolerance").as_double();
  m_base_pose = Eigen::Map<const Eigen::Matrix<double, 3, 4>>(
      this->get_parameter("base_pose").as_double_array().data());
  m_pose_wc = Eigen::Map<const Eigen::Matrix<double, 3, 4>>(
      this->get_parameter("pose_wc").as_double_array().data());
  m_finger_offset = Eigen::Map<const Eigen::Vector3d>(
      this->get_parameter("finger_offset").as_double_array().data());
  auto finger_ids = this->get_parameter("finger_ids").as_integer_array();
  m_finger_ids.resize(finger_ids.size());
  std::copy(finger_ids.begin(), finger_ids.end(), m_finger_ids.begin());

  const std::string color_topic =
      this->get_parameter("color_topic").as_string();
  const std::string depth_topic =
      this->get_parameter("depth_topic").as_string();
  const std::string joint_state_topic =
      this->get_parameter("joint_state_topic").as_string();
  const std::string sam_topic =
      this->get_parameter("segmentation_topic").as_string();
  const std::string meta_hand_urdf =
      this->get_parameter("meta_hand_urdf").as_string();
  const std::string model_path = this->get_parameter("model_path").as_string();
  const std::string sam_encoder_checkpoint =
      model_path + this->get_parameter("sam_encoder").as_string();
  const std::string sam_decoder_checkpoint =
      model_path + this->get_parameter("sam_decoder").as_string();
  const std::string mobile_sam_encoder_checkpoint =
      model_path + this->get_parameter("mobile_sam_encoder").as_string();
  const std::string mobile_sam_decoder_checkpoint =
      model_path + this->get_parameter("mobile_sam_decoder").as_string();
  const std::string superpoint_checkpoint =
      model_path + this->get_parameter("superpoint").as_string();
  const std::string lightglue_checkpoint =
      model_path + this->get_parameter("lightglue").as_string();
  const std::string ostrack_checkpoint =
      model_path + this->get_parameter("ostrack").as_string();
  const std::string trt_engine_cache_path =
      model_path + this->get_parameter("trt_engine_cache").as_string();

  using std::placeholders::_1;
  using std::placeholders::_2;
  using std::placeholders::_3;

  m_segmentation_publisher =
      this->create_publisher<sensor_msgs::msg::Image>(sam_topic, 10);
  m_color_subscriber = std::make_shared<
      message_filters::Subscriber<sensor_msgs::msg::CompressedImage>>(
      this, color_topic);
  m_depth_subscriber =
      std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(
          this, depth_topic);
  m_joint_subscriber = std::make_shared<
      message_filters::Subscriber<sensor_msgs::msg::JointState>>(
      this, joint_state_topic);
  m_synchronizer =
      std::make_shared<Synchronizer>(ApproximatePolicy(10), *m_color_subscriber,
                                     *m_depth_subscriber, *m_joint_subscriber);
  m_synchronizer->registerCallback(
      std::bind(&SAMPublisher::CallBack, this, _1, _2, _3));

  igraph_rng_seed(igraph_rng_default(), 0);

  CHECK_CUDA(cudaSetDevice(m_device));
  m_handle = std::make_shared<gum::graph::Handle>();

  m_sam = std::make_shared<gum::perception::segmentation::SAM>(
      sam_encoder_checkpoint, sam_decoder_checkpoint, m_device);
  m_mobile_sam = std::make_shared<gum::perception::segmentation::SAM>(
      mobile_sam_encoder_checkpoint, mobile_sam_decoder_checkpoint, m_device);
  m_superpoint = std::make_shared<gum::perception::feature::SuperPoint>(
      superpoint_checkpoint, trt_engine_cache_path, m_device);
  m_lightglue = std::make_shared<gum::perception::feature::LightGlue>(
      lightglue_checkpoint, trt_engine_cache_path, m_device);
  m_ostracker = std::make_shared<gum::perception::bbox::OSTrack>(
      ostrack_checkpoint, trt_engine_cache_path, m_device);

  m_realsense = std::make_shared<gum::perception::dataset::RealSenseDataset<
      gum::perception::dataset::Device::GPU>>(
      m_device, m_height, m_width, m_intrinsics[0], m_intrinsics[1],
      m_intrinsics[2], m_intrinsics[3], m_intrinsics[4], m_intrinsics[5],
      m_intrinsics[6], m_intrinsics[7], m_depth_scale);

  pinocchio::urdf::buildModel(meta_hand_urdf, m_robot_model);

  WarmUp();
  RCLCPP_INFO_STREAM(this->get_logger(),
                     "-------------------------------------------------");
  RCLCPP_INFO_STREAM(this->get_logger(), "GUM frontend has been setup.");
  RCLCPP_INFO_STREAM(this->get_logger(),
                     "-------------------------------------------------");
}

void SAMPublisher::WarmUp() {
  RCLCPP_INFO_STREAM(this->get_logger(),
                     "-------------------------------------------------");
  RCLCPP_INFO_STREAM(this->get_logger(), "GUM frontend starts to warm up");
  RCLCPP_INFO_STREAM(this->get_logger(),
                     "-------------------------------------------------");
  const std::string test_image_path =
      this->get_parameter("test_image").as_string();

  cv::Mat test_image = cv::imread(test_image_path, cv::IMREAD_UNCHANGED);
  cv::cvtColor(test_image, test_image, cv::COLOR_BGR2RGB);

  int height = test_image.rows;
  int width = test_image.cols;

  std::vector<Eigen::Vector2f> point_coords_v{{0.5f * width, 0.5f * height}};
  Eigen::Vector4f bbox = {point_coords_v[0][0] - 0.2f * width,
                          point_coords_v[0][1] - 0.2f * height,
                          point_coords_v[0][0] + 0.2f * width,
                          point_coords_v[0][1] + 0.2f * height};
  std::vector<float> point_labels_v(point_coords_v.size(), 1.0f);
  torch::Tensor masks, scores, logits;
  for (int i = 0; i < 5; i++) {
    m_mobile_sam->SetImage(test_image);
    m_mobile_sam->Query(point_coords_v, point_labels_v, bbox, masks, scores,
                        logits);
    m_sam->SetImage(test_image);
    m_sam->Query(point_coords_v, point_labels_v, bbox, masks, scores, logits);
  }

  m_ostracker->Initialize(test_image, bbox);
  for (int i = 0; i < 10; i++) {
    Eigen::Vector4f target_bbox;
    m_ostracker->Track(test_image, bbox, target_bbox);
    m_mobile_sam->SetImage(test_image);
    m_mobile_sam->Query(point_coords_v, point_labels_v, bbox, masks, scores,
                        logits);

    cv::Mat cropped_image =
        test_image(cv::Range(bbox[1], bbox[3]), cv::Range(bbox[0], bbox[2]));
    cv::cvtColor(cropped_image, cropped_image, cv::COLOR_RGB2GRAY);
    m_superpoint->Extract(cropped_image, m_initial_keypoints_v,
                          m_initial_normalized_keypoints_v,
                          m_initial_keypoint_scores_v, m_initial_descriptors_v);

    std::vector<float> match_scores_v;
    std::vector<Eigen::Vector2i> initial_matches_v;
    m_lightglue->Match(m_initial_normalized_keypoints_v,
                       m_initial_normalized_keypoints_v,
                       m_initial_descriptors_v, m_initial_descriptors_v,
                       initial_matches_v, match_scores_v);
  }
}

void SAMPublisher::Initialize(const cv::Mat &image, const cv::Mat &depth,
                              const Eigen::VectorXd &joint_angles) {
  Frame curr_frame;
  curr_frame.id = 0;
  curr_frame.image = image;
  curr_frame.depth = depth;

  std::vector<Eigen::Vector3d> finger_tips;
  Eigen::Vector2d pixel_c;

  GetFingerTips(joint_angles, finger_tips);
  ProjectGraspCenter(finger_tips, pixel_c);

  m_sam->SetImage(curr_frame.image);
  std::vector<Eigen::Vector2f> point_coords_v{pixel_c.cast<float>()};
  std::vector<float> point_labels_v(point_coords_v.size(), 1.0f);
  torch::Tensor masks, scores, logits;
  m_sam->Query(point_coords_v, point_labels_v, torch::nullopt, masks, scores,
               logits);

  float target_area = 0.03 * m_width * m_height;
  masks = masks[0].to(torch::kUInt8);
  int sel = (masks.sum({1, 2}).to(torch::kCPU) - target_area)
                .abs()
                .argmin()
                .item()
                .toInt();

  curr_frame.mask_gpu = masks[sel];
  curr_frame.mask_cpu = masks[sel].to(torch::kCPU);

  auto orig_mask = masks[sel].to(torch::kInt16);
  gum::perception::utils::GetBox(m_height, m_width,
                                 (uint16_t *)orig_mask.data_ptr<int16_t>(),
                                 curr_frame.bbox, m_handle->GetStream());
  gum::perception::utils::RefineMask(m_height, m_width, curr_frame.bbox,
                                     curr_frame.mask_cpu.data_ptr<uint8_t>());
  curr_frame.mask_gpu = curr_frame.mask_cpu.to(curr_frame.mask_gpu.device());
  curr_frame.offset = curr_frame.bbox.head<2>().cast<float>();

  m_ostracker->Initialize(curr_frame.image, curr_frame.bbox.cast<float>());
  ExtractKeyPoints(curr_frame, curr_frame.mask_cpu.data_ptr<uint8_t>());
  WriteFrame(curr_frame);
  m_frames_v.push_back(std::move(curr_frame));
}

void SAMPublisher::Process(const cv::Mat &image, const cv::Mat &depth,
                           const Eigen::VectorXd &joint_angles) {
  const auto &prev_frame = m_frames_v.back();
  Frame curr_frame;
  curr_frame.id = prev_frame.id + 1;
  curr_frame.image = image;
  curr_frame.depth = depth;

  Eigen::Vector4f init_bbox;
  m_ostracker->Track(curr_frame.image, prev_frame.bbox.cast<float>(),
                     init_bbox);
  curr_frame.bbox = (init_bbox.array() + 0.5f).cast<int>();
  curr_frame.offset = curr_frame.bbox.head<2>().cast<float>();

  torch::Tensor extended_mask_cpu = torch::empty(
      prev_frame.mask_cpu.sizes(),
      torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
  int extended_radius = std::round(
      std::max(1.f, 0.080f * std::sqrt((init_bbox[3] - init_bbox[1]) *
                                       (init_bbox[2] - init_bbox[0]))));
  int shrinked_radius = 1;
  gum::perception::utils::ExtendMasks(m_height, m_width, curr_frame.bbox,
                                      prev_frame.mask_cpu.data_ptr<uint8_t>(),
                                      extended_mask_cpu.data_ptr<uint8_t>(),
                                      extended_radius);

  torch::Tensor shrinked_mask_cpu = torch::empty(
      prev_frame.mask_cpu.sizes(),
      torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
  gum::perception::utils::ShrinkMasks(m_height, m_width, prev_frame.bbox,
                                      prev_frame.mask_cpu.data_ptr<uint8_t>(),
                                      shrinked_mask_cpu.data_ptr<uint8_t>(),
                                      shrinked_radius);

  RCLCPP_INFO_STREAM(this->get_logger(),
                     "Frame " << curr_frame.id
                              << ": SuperPoint has extracted keypoints.");
  ExtractKeyPoints(curr_frame, extended_mask_cpu.data_ptr<uint8_t>());
#if 0
  cv::Mat cropped_image;
  curr_frame
      .image(cv::Range(curr_frame.bbox[1], curr_frame.bbox[3]),
             cv::Range(curr_frame.bbox[0], curr_frame.bbox[2]))
      .copyTo(cropped_image,
              cv::Mat(curr_frame.image.size(), CV_8U,
                      extended_mask_cpu.data_ptr<uint8_t>())(
                  cv::Range(curr_frame.bbox[1], curr_frame.bbox[3]),
                  cv::Range(curr_frame.bbox[0], curr_frame.bbox[2])));
  // Initial Mask
  cv::cvtColor(cropped_image, cropped_image, cv::COLOR_RGB2GRAY);
  // Extract Keypoints
  m_superpoint->Extract(cropped_image, m_initial_keypoints_v,
                        m_initial_normalized_keypoints_v,
                        m_initial_keypoint_scores_v, m_initial_descriptors_v);
  for (auto &initial_keypoint : m_initial_keypoints_v) {
    initial_keypoint += curr_frame.offset;
  }

  int num_initial_keypoints = m_initial_keypoints_v.size();
  gum::perception::utils::SelectKeyPointsByDepth(
      num_initial_keypoints, m_min_depth, m_max_depth, m_depth_scale,
      curr_frame.depth, m_initial_keypoints_v, m_initial_descriptors_v,
      m_initial_normalized_keypoints_v, curr_frame.keypoints_v,
      curr_frame.descriptors_v, curr_frame.normalized_keypoints_v);

  // Point Clouds
  int num_keypoints = curr_frame.keypoints_v.size();
  gum::perception::utils::GetPointClouds(
      num_keypoints, m_intrinsics[0], m_intrinsics[1], m_intrinsics[2],
      m_intrinsics[3], m_depth_scale, curr_frame.depth, curr_frame.keypoints_v,
      curr_frame.point_clouds_v);
#endif

  // Feature Matching
  thrust::device_vector<int> d_initial_matches_v;
  std::vector<float> match_scores_v;
  std::vector<Eigen::Vector2i> initial_matches_v;
  m_lightglue->Match(prev_frame.normalized_keypoints_v,
                     curr_frame.normalized_keypoints_v,
                     prev_frame.descriptors_v, curr_frame.descriptors_v,
                     initial_matches_v, match_scores_v);
  int num_initial_matches = initial_matches_v.size();
  RCLCPP_INFO_STREAM(this->get_logger(),
                     "Frame " << curr_frame.id << ": LightGlue has matched "
                              << num_initial_matches << " pairs of keypoints.");
  d_initial_matches_v.resize(2 * num_initial_matches);
  gum::utils::HostArrayOfMatrixToDeviceMatrixOfArray(initial_matches_v,
                                                     d_initial_matches_v);
  thrust::device_vector<int> d_matches_v;
  gum::perception::feature::RejectOutliers(
      *m_handle, m_graph_params, m_leiden_params, m_outlier_tolerance,
      prev_frame.point_clouds_v.size() / 3,
      curr_frame.point_clouds_v.size() / 3, num_initial_matches,
      prev_frame.point_clouds_v, curr_frame.point_clouds_v, d_initial_matches_v,
      d_matches_v);
  int num_matches = d_matches_v.size() / 2;
  RCLCPP_INFO_STREAM(this->get_logger(),
                     "Frame " << curr_frame.id << ": Outlier rejection keeps "
                              << num_matches << "/" << num_initial_matches
                              << " pairs of keypoints.");
  Eigen::Matrix<float, 3, 4> relative_pose;
  gum::perception::feature::EstimateRelativePose(
      *m_handle, prev_frame.point_clouds_v.size() / 3,
      curr_frame.point_clouds_v.size() / 3, num_matches,
      prev_frame.point_clouds_v, curr_frame.point_clouds_v, d_matches_v,
      relative_pose);

  // Segmentation
  std::vector<Eigen::Vector2i> matches_v(num_matches);
  gum::utils::DeviceMatrixOfArrayToHostArrayOfMatrix(d_matches_v, matches_v);
  std::vector<int> selected_match_indices_v;
  std::vector<Eigen::Vector2f> point_coords_v;
  gum::perception::utils::SelectMatchesForSAM(
      m_height, m_width, prev_frame.keypoints_v.size(),
      curr_frame.keypoints_v.size(), num_matches, prev_frame.bbox.cast<float>(),
      curr_frame.bbox.cast<float>(), prev_frame.mask_cpu.data_ptr<uint8_t>(),
      prev_frame.mask_cpu.data_ptr<uint8_t>(), prev_frame.keypoints_v,
      curr_frame.keypoints_v, matches_v, selected_match_indices_v,
      point_coords_v);
  std::vector<float> point_labels_v(point_coords_v.size(), 1.0f);

  torch::Tensor masks, scores, logits;
  m_mobile_sam->SetImage(curr_frame.image);
  m_mobile_sam->Query(point_coords_v, point_labels_v, init_bbox, masks, scores,
                      logits);
  RCLCPP_INFO_STREAM(this->get_logger(),
                     "Frame " << curr_frame.id << ": SAM has segmented image.");
  curr_frame.mask_gpu = masks[0][1].to(torch::kUInt8);
  curr_frame.mask_cpu = curr_frame.mask_gpu.to(torch::kCPU);
  gum::perception::utils::FilterMaskByDepth(
      m_height, m_width, curr_frame.bbox, m_min_depth, m_max_depth,
      m_depth_scale, curr_frame.depth, curr_frame.mask_cpu.data_ptr<uint8_t>());
  gum::perception::utils::RefineMask(m_height, m_width, curr_frame.bbox,
                                     curr_frame.mask_cpu.data_ptr<uint8_t>());
  curr_frame.mask_gpu = curr_frame.mask_cpu.to(curr_frame.mask_gpu.device());

  RCLCPP_INFO_STREAM(this->get_logger(),
                     "Frame " << curr_frame.id
                              << ": Segmentation has been refined.");
  // Refine Keypoints
  int num_initial_keypoints = curr_frame.keypoints_v.size();
  RefineKeyPoints(curr_frame);
  int num_keypoints = curr_frame.keypoints_v.size();
  RCLCPP_INFO_STREAM(this->get_logger(), "Frame "
                                             << curr_frame.id
                                             << ": Keypoints has been refined ("
                                             << num_keypoints << "/"
                                             << num_initial_keypoints << ").");
  WriteFrame(curr_frame);
  m_frames_v.push_back(std::move(curr_frame));
}

void SAMPublisher::AddFrame(
    const sensor_msgs::msg::CompressedImage::ConstSharedPtr &color_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr &depth_msg,
    const sensor_msgs::msg::JointState::ConstSharedPtr &joint_msg) {
  cv_bridge::CvImagePtr color_ptr, depth_ptr;
  color_ptr =
      cv_bridge::toCvCopy(color_msg, sensor_msgs::image_encodings::BGR8);
  depth_ptr = cv_bridge::toCvCopy(depth_msg, "16UC1");

  double timestamp = double(color_msg->header.stamp.sec) +
                     1e-9 * double(color_msg->header.stamp.nanosec);
  m_realsense->AddFrame(
      {timestamp, std::move(color_ptr->image), std::move(depth_ptr->image)});
  Eigen::Map<const Eigen::VectorXd> raw_joint_angles(
      joint_msg->position.data(), joint_msg->position.size());
  Eigen::VectorXd joint_angles(m_robot_model.nq);
  joint_angles.segment<4>(0) = raw_joint_angles.segment<4>(0);
  joint_angles.segment<4>(4) = raw_joint_angles.segment<4>(12);
  joint_angles.segment<8>(8) = raw_joint_angles.segment<8>(4);
  m_joint_angles_v.push_back(std::move(joint_angles));
}

void SAMPublisher::ProjectGraspCenter(
    const std::vector<Eigen::Vector3d> &finger_tips,
    Eigen::Vector2d &grasp_center) {
  Eigen::Vector3d point_w = Eigen::Vector3d::Zero();
  for (const auto &finger_tip : finger_tips) {
    point_w += finger_tip;
  }

  point_w /= finger_tips.size();
  point_w[2] += 0.03;

  Eigen::Vector3d point_c =
      m_pose_wc.leftCols<3>().transpose() * (point_w - m_pose_wc.col(3));

  grasp_center = point_c.head<2>() / point_c[2];
  grasp_center[0] = -m_intrinsics[0] * grasp_center[0] + m_intrinsics[2];
  grasp_center[1] = m_intrinsics[1] * grasp_center[1] + m_intrinsics[3];
}

void SAMPublisher::ExtractKeyPoints(Frame &curr_frame,
                                    const uint8_t *mask_ptr) {
  // Feature Extraction
  cv::Mat cropped_image;
  const cv::Mat mask(curr_frame.image.size(), CV_8U, (void *)mask_ptr);
  curr_frame
      .image(cv::Range(curr_frame.bbox[1], curr_frame.bbox[3]),
             cv::Range(curr_frame.bbox[0], curr_frame.bbox[2]))
      .copyTo(cropped_image,
              mask(cv::Range(curr_frame.bbox[1], curr_frame.bbox[3]),
                   cv::Range(curr_frame.bbox[0], curr_frame.bbox[2])));
  cv::cvtColor(cropped_image, cropped_image, cv::COLOR_RGB2GRAY);
  m_superpoint->Extract(cropped_image, m_initial_keypoints_v,
                        m_initial_normalized_keypoints_v,
                        m_initial_keypoint_scores_v, m_initial_descriptors_v);

  for (auto &initial_keypoint : m_initial_keypoints_v) {
    initial_keypoint += curr_frame.offset;
  }
  int num_initial_keypoints = m_initial_keypoints_v.size();
  gum::perception::utils::SelectKeyPointsByDepth(
      num_initial_keypoints, m_min_depth, m_max_depth, m_depth_scale,
      curr_frame.depth, m_initial_keypoints_v, m_initial_descriptors_v,
      m_initial_normalized_keypoints_v, curr_frame.keypoints_v,
      curr_frame.descriptors_v, curr_frame.normalized_keypoints_v);
  gum::perception::utils::GetPointClouds(
      curr_frame.keypoints_v.size(), m_intrinsics[0], m_intrinsics[1],
      m_intrinsics[2], m_intrinsics[3], m_depth_scale, curr_frame.depth,
      curr_frame.keypoints_v, curr_frame.point_clouds_v);
}

void SAMPublisher::RefineKeyPoints(Frame &curr_frame) {
  auto is_valid = curr_frame.mask_cpu.data_ptr<uint8_t>();
  int num_keypoints = curr_frame.keypoints_v.size();
  m_initial_keypoints_v.clear();
  m_initial_descriptors_v.clear();
  m_initial_normalized_keypoints_v.clear();

  for (int n = 0; n < num_keypoints; n++) {
    const auto &keypoint = curr_frame.keypoints_v[n];
    int x = std::round(keypoint[0]);
    int y = std::round(keypoint[1]);
    if (is_valid[y * m_width + x]) {
      m_initial_keypoints_v.push_back(curr_frame.keypoints_v[n]);
      m_initial_descriptors_v.push_back(curr_frame.descriptors_v[n]);
      m_initial_normalized_keypoints_v.push_back(
          curr_frame.normalized_keypoints_v[n]);
    }
  }
  std::swap(m_initial_keypoints_v, curr_frame.keypoints_v);
  std::swap(m_initial_descriptors_v, curr_frame.descriptors_v);
  std::swap(m_initial_normalized_keypoints_v,
            curr_frame.normalized_keypoints_v);

  gum::perception::utils::GetPointClouds(
      curr_frame.keypoints_v.size(), m_intrinsics[0], m_intrinsics[1],
      m_intrinsics[2], m_intrinsics[3], m_depth_scale, curr_frame.depth,
      curr_frame.keypoints_v, curr_frame.point_clouds_v);
}

void SAMPublisher::WriteFrame(const Frame &frame) {
  cv::Mat image;
  cv::cvtColor(frame.image, image, CV_RGB2BGR);

  cv::Mat masked_image;
  image.copyTo(masked_image, cv::Mat(image.size(), CV_8U,
                                     frame.mask_cpu.data_ptr<uint8_t>()));
  cv::imwrite("image_" + std::to_string(frame.id) + "_mask.jpg", masked_image);

  std::vector<cv::KeyPoint> cv_keypoints_v;
  for (const auto &keypoint : frame.keypoints_v) {
    cv_keypoints_v.push_back({keypoint[0], keypoint[1], 1});
  }

  cv::Mat masked_image_with_keypoints;
  cv::drawKeypoints(masked_image, cv_keypoints_v, masked_image_with_keypoints,
                    cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
  cv::imwrite("image_" + std::to_string(frame.id) + "_keypoints.jpg",
              masked_image_with_keypoints);

  cv::rectangle(image, cv::Point(frame.bbox[0], frame.bbox[1]),
                cv::Point(frame.bbox[2], frame.bbox[3]), cv::Scalar(0, 0, 255),
                1, cv::LINE_8);
  cv::imwrite("image_" + std::to_string(frame.id) + "_bbox.jpg", image);
}

void SAMPublisher::GetFingerTips(const Eigen::VectorXd &joint_angles,
                                 std::vector<Eigen::Vector3d> &finger_tips) {
  gum::utils::GetFingerTips(m_robot_model, joint_angles, m_base_pose,
                            m_finger_offset, m_finger_ids, finger_tips);
}

void SAMPublisher::CallBack(
    const sensor_msgs::msg::CompressedImage::ConstSharedPtr &color_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr &depth_msg,
    const sensor_msgs::msg::JointState::ConstSharedPtr &joint_msg) {
#if 1
  this->AddFrame(color_msg, depth_msg, joint_msg);
  RCLCPP_INFO_STREAM(
      this->get_logger(),
      "=========================================================");
  RCLCPP_INFO_STREAM(this->get_logger(), "Frame " << m_frames_v.size());
  RCLCPP_INFO_STREAM(
      this->get_logger(),
      "---------------------------------------------------------");
  if (m_frames_v.size() == 0) {
    Initialize(m_realsense->GetFrames().back().image,
               m_realsense->GetFrames().back().depth, m_joint_angles_v.back());
  } else {
    Process(m_realsense->GetFrames().back().image,
            m_realsense->GetFrames().back().depth, m_joint_angles_v.back());
  }
  //   if (m_realsense->GetNumFrames() >= 1000) {
  //     m_realsense->Clear();
  //   }
  //   if (m_joint_angles_v.size() >= 1000) {
  //     m_joint_angles_v.clear();
  //   }
#else
  RCLCPP_INFO(this->get_logger(), "I heard: '%d', '%d' and '%f'",
              (int)color_msg->data.size(), (int)depth_msg->data.size(),
              double(joint_msg->header.stamp.sec));
#endif
}
} // namespace perception
} // namespace gum