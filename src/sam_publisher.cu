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
template <typename ColorMsg, typename DepthMsg>
SAMPublisher<ColorMsg, DepthMsg>::SAMPublisher(const std::string &node_name)
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
  this->declare_parameter("sam_offset", rclcpp::PARAMETER_DOUBLE);
  this->declare_parameter("finger_ids", rclcpp::PARAMETER_INTEGER_ARRAY);
  this->declare_parameter("save_results", rclcpp::PARAMETER_INTEGER);
  this->declare_parameter("result_path", rclcpp::PARAMETER_STRING);

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
  m_sam_offset = this->get_parameter("sam_offset").as_double();
  auto finger_ids = this->get_parameter("finger_ids").as_integer_array();
  m_finger_ids.resize(finger_ids.size());
  std::copy(finger_ids.begin(), finger_ids.end(), m_finger_ids.begin());
  m_save_results = this->get_parameter("save_results").as_int();
  m_result_path = this->get_parameter("result_path").as_string();

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

  m_seg_depth_publisher = this->create_publisher<ImageMsg>(sam_topic, 10);
  m_color_subscriber = std::make_shared<message_filters::Subscriber<ColorMsg>>(
      this, color_topic);
  m_depth_subscriber = std::make_shared<message_filters::Subscriber<DepthMsg>>(
      this, depth_topic);
  m_joint_subscriber = std::make_shared<message_filters::Subscriber<JointMsg>>(
      this, joint_state_topic);
  m_synchronizer =
      std::make_shared<Synchronizer>(ApproximatePolicy(20), *m_color_subscriber,
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
  m_superpoint = std::make_shared<gum::perception::feature::FastSuperPoint>(
      superpoint_checkpoint, trt_engine_cache_path, m_device);
  m_lightglue = std::make_shared<gum::perception::feature::LightGlue>(
      lightglue_checkpoint, trt_engine_cache_path, m_device);
  m_ostracker = std::make_shared<gum::perception::bbox::OSTrack>(
      ostrack_checkpoint, trt_engine_cache_path, m_device);

  m_realsense_camera =
      std::make_shared<gum::perception::camera::RealSenseCamera<
          gum::perception::camera::Device::GPU>>(
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

  m_num_frames = 0;
}

template <typename ColorMsg, typename DepthMsg>
void SAMPublisher<ColorMsg, DepthMsg>::Reset() const {
  m_frames_v.clear();
  m_joint_angles_v.clear();
  m_num_frames = 0;
}

template <typename ColorMsg, typename DepthMsg>
void SAMPublisher<ColorMsg, DepthMsg>::Clear() const {
  this->Reset();
  m_num_frames = 0;
}

template <typename ColorMsg, typename DepthMsg>
void SAMPublisher<ColorMsg, DepthMsg>::Initialize(
    const cv::Mat &image, const cv::Mat &depth,
    const Eigen::VectorXd &joint_angles, FramePtr curr_frame) const {
  curr_frame->id = m_num_frames;
  curr_frame->image = image;
  curr_frame->depth = depth;

  std::vector<Eigen::Vector3d> finger_tips;
  Eigen::Vector2d pixel_c;

  std::vector<Eigen::Vector2d> fingers_c;
  GetFingerTips(joint_angles, finger_tips);
  ProjectGraspCenter(finger_tips, fingers_c, pixel_c);
  RCLCPP_INFO_STREAM(this->get_logger(),
                     "Frame " << curr_frame->id
                              << ": Grasp center has been computed.");
  if (m_save_results >= 1) {
    cv::Mat image;
    cv::cvtColor(curr_frame->image, image, CV_RGB2BGR);
    std::vector<cv::KeyPoint> cv_keypoints_v;
    for (const auto &keypoint : fingers_c) {
      cv_keypoints_v.push_back({(float)keypoint[0], (float)keypoint[1], 1});
    }
    cv_keypoints_v.push_back({(float)pixel_c[0], (float)pixel_c[1], 1});
    cv::Mat image_with_tips;
    cv::drawKeypoints(image, cv_keypoints_v, image_with_tips,
                      cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
    cv::imwrite(m_result_path + "image_" + std::to_string(curr_frame->id) +
                    "_tips.jpg",
                image_with_tips);
  }

  m_sam->SetImage(curr_frame->image);
  std::vector<Eigen::Vector2f> point_coords_v{pixel_c.cast<float>()};
  std::vector<float> point_labels_v(point_coords_v.size(), 1.0f);
  torch::Tensor masks, scores, logits;
  m_sam->Query(point_coords_v, point_labels_v, torch::nullopt, masks, scores,
               logits);
  RCLCPP_INFO_STREAM(this->get_logger(), "Frame "
                                             << curr_frame->id
                                             << ": Initial segmentation done.");

  float target_area = 0.03 * m_width * m_height;
  masks = masks[0].to(torch::kUInt8);
  int sel = (masks.sum({1, 2}).to(torch::kCPU) - target_area)
                .abs()
                .argmin()
                .item()
                .toInt();

  curr_frame->mask_gpu = masks[sel];
  curr_frame->mask_cpu = masks[sel].to(torch::kCPU);

  if (m_save_results >= 1) {
    for (int i = 0; i < 3; i++) {
      const cv::Mat mask(image.size(), CV_8U,
                         masks[i].to(torch::kCPU).data_ptr<uint8_t>());
      cv::Mat masked_image;
      image.copyTo(masked_image, mask);
      cv::cvtColor(masked_image, masked_image, CV_RGB2BGR);
      cv::imwrite(m_result_path + "image_" + std::to_string(0) + "_masked_" +
                      std::to_string(i) + ".jpg",
                  masked_image);
    }
  }

  auto orig_mask = masks[sel].to(torch::kInt16);
  gum::perception::utils::GetBox(m_height, m_width,
                                 (uint16_t *)orig_mask.data_ptr<int16_t>(),
                                 curr_frame->bbox, m_handle->GetStream());
  gum::perception::utils::RefineMask(m_height, m_width, curr_frame->bbox,
                                     curr_frame->mask_cpu.data_ptr<uint8_t>());
  gum::perception::utils::GetBox(m_height, m_width,
                                 (uint16_t *)orig_mask.data_ptr<int16_t>(),
                                 curr_frame->bbox, m_handle->GetStream());
  curr_frame->mask_gpu = curr_frame->mask_cpu.to(curr_frame->mask_gpu.device());
  curr_frame->offset = curr_frame->bbox.head<2>().cast<float>();

  m_ostracker->Initialize(curr_frame->image, curr_frame->bbox.cast<float>());
  RCLCPP_INFO_STREAM(this->get_logger(),
                     "Frame " << curr_frame->id
                              << ": Bounding box has been created.");
  ExtractKeyPoints(curr_frame, curr_frame->mask_cpu.data_ptr<uint8_t>());
  RCLCPP_INFO_STREAM(this->get_logger(),
                     "Frame " << curr_frame->id << ": SuperPoint has extracted "
                              << curr_frame->keypoints_v.size()
                              << " keypoints.");
  if (m_save_results >= 1) {
    WriteFrame(curr_frame);
  }
}

template <typename ColorMsg, typename DepthMsg>
void SAMPublisher<ColorMsg, DepthMsg>::Iterate(
    const cv::Mat &image, const cv::Mat &depth,
    const Eigen::VectorXd &joint_angles, FrameConstPtr prev_frame,
    FramePtr curr_frame) const {
  curr_frame->id = m_num_frames;
  curr_frame->image = image;
  curr_frame->depth = depth;

  Eigen::Vector4f init_bbox;
  m_ostracker->Track(curr_frame->image, prev_frame->bbox.cast<float>(),
                     init_bbox);
  curr_frame->bbox = (init_bbox.array() + 0.5f).cast<int>();
  curr_frame->offset = curr_frame->bbox.head<2>().cast<float>();

  torch::Tensor extended_mask_cpu = torch::empty(
      prev_frame->mask_cpu.sizes(),
      torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
  int extended_radius = std::round(
      std::max(1.f, 0.080f * std::sqrt((init_bbox[3] - init_bbox[1]) *
                                       (init_bbox[2] - init_bbox[0]))));
  gum::perception::utils::ExtendMasks(m_height, m_width, curr_frame->bbox,
                                      prev_frame->mask_cpu.data_ptr<uint8_t>(),
                                      extended_mask_cpu.data_ptr<uint8_t>(),
                                      extended_radius);

  ExtractKeyPoints(curr_frame, extended_mask_cpu.data_ptr<uint8_t>());
  RCLCPP_INFO_STREAM(this->get_logger(),
                     "Frame " << curr_frame->id << ": SuperPoint has extracted "
                              << curr_frame->keypoints_v.size()
                              << " keypoints.");
  // Feature Matching
  thrust::device_vector<int> d_initial_matches_v;
  std::vector<float> match_scores_v;
  std::vector<Eigen::Vector2i> initial_matches_v;
  m_lightglue->Match(prev_frame->normalized_keypoints_v,
                     curr_frame->normalized_keypoints_v,
                     prev_frame->descriptors_v, curr_frame->descriptors_v,
                     initial_matches_v, match_scores_v);
  int num_initial_matches = initial_matches_v.size();
  RCLCPP_INFO_STREAM(this->get_logger(),
                     "Frame " << curr_frame->id << ": LightGlue has matched "
                              << num_initial_matches << " pairs of keypoints.");
  d_initial_matches_v.resize(2 * num_initial_matches);
  gum::utils::HostArrayOfMatrixToDeviceMatrixOfArray(initial_matches_v,
                                                     d_initial_matches_v);
  thrust::device_vector<int> d_matches_v;
  gum::perception::feature::RejectOutliers(
      *m_handle, m_graph_params, m_leiden_params, m_outlier_tolerance,
      prev_frame->point_clouds_v.size() / 3,
      curr_frame->point_clouds_v.size() / 3, num_initial_matches,
      prev_frame->point_clouds_v, curr_frame->point_clouds_v,
      d_initial_matches_v, d_matches_v);
  int num_matches = d_matches_v.size() / 2;
  RCLCPP_INFO_STREAM(this->get_logger(),
                     "Frame " << curr_frame->id << ": Outlier rejection keeps "
                              << num_matches << "/" << num_initial_matches
                              << " pairs of keypoints.");
  Eigen::Matrix<float, 3, 4> relative_pose;
  gum::perception::feature::EstimateRelativePose(
      *m_handle, prev_frame->point_clouds_v.size() / 3,
      curr_frame->point_clouds_v.size() / 3, num_matches,
      prev_frame->point_clouds_v, curr_frame->point_clouds_v, d_matches_v,
      relative_pose);

  // Segmentation
  std::vector<Eigen::Vector2i> matches_v(num_matches);
  gum::utils::DeviceMatrixOfArrayToHostArrayOfMatrix(d_matches_v, matches_v);
  std::vector<int> selected_match_indices_v;
  std::vector<Eigen::Vector2f> point_coords_v;
  gum::perception::utils::SelectMatchesForSAM(
      m_height, m_width, prev_frame->keypoints_v.size(),
      curr_frame->keypoints_v.size(), num_matches,
      prev_frame->bbox.cast<float>(), curr_frame->bbox.cast<float>(),
      prev_frame->mask_cpu.data_ptr<uint8_t>(),
      prev_frame->mask_cpu.data_ptr<uint8_t>(), prev_frame->keypoints_v,
      curr_frame->keypoints_v, matches_v, selected_match_indices_v,
      point_coords_v);
  std::vector<float> point_labels_v(point_coords_v.size(), 1.0f);

  torch::Tensor masks, scores, logits;
  m_mobile_sam->SetImage(curr_frame->image);
  m_mobile_sam->Query(point_coords_v, point_labels_v, init_bbox, masks, scores,
                      logits);
  RCLCPP_INFO_STREAM(this->get_logger(), "Frame "
                                             << curr_frame->id
                                             << ": SAM has segmented image.");
  curr_frame->mask_gpu = masks[0][1].to(torch::kUInt8);
  curr_frame->mask_cpu = curr_frame->mask_gpu.to(torch::kCPU);
  gum::perception::utils::FilterMaskByDepth(
      m_height, m_width, curr_frame->bbox, m_min_depth, m_max_depth,
      m_depth_scale, curr_frame->depth,
      curr_frame->mask_cpu.data_ptr<uint8_t>());
  gum::perception::utils::RefineMask(m_height, m_width, curr_frame->bbox,
                                     curr_frame->mask_cpu.data_ptr<uint8_t>());
  curr_frame->mask_gpu = curr_frame->mask_cpu.to(curr_frame->mask_gpu.device());

  RCLCPP_INFO_STREAM(this->get_logger(),
                     "Frame " << curr_frame->id
                              << ": Segmentation has been refined.");
  if (m_save_results >= 2) {
    WriteMatch(prev_frame, curr_frame, matches_v);
  }

  // Refine Keypoints
  int num_initial_keypoints = curr_frame->keypoints_v.size();
  RefineKeyPoints(curr_frame);
  int num_keypoints = curr_frame->keypoints_v.size();
  RCLCPP_INFO_STREAM(this->get_logger(), "Frame "
                                             << curr_frame->id
                                             << ": Keypoints has been refined ("
                                             << num_keypoints << "/"
                                             << num_initial_keypoints << ").");
  if (m_save_results >= 1) {
    WriteFrame(curr_frame);
  }
}

template <typename ColorMsg, typename DepthMsg>
void SAMPublisher<ColorMsg, DepthMsg>::WarmUp() const {
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
  for (int i = 0; i < 2; i++) {
    m_mobile_sam->SetImage(test_image);
    m_mobile_sam->Query(point_coords_v, point_labels_v, bbox, masks, scores,
                        logits);
    m_sam->SetImage(test_image);
    m_sam->Query(point_coords_v, point_labels_v, bbox, masks, scores, logits);
  }

  m_ostracker->Initialize(test_image, bbox);
  for (int i = 0; i < 5; i++) {
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

template <typename ColorMsg, typename DepthMsg>
void SAMPublisher<ColorMsg, DepthMsg>::AddFrame(
    typename ColorMsg::ConstSharedPtr color_msg,
    typename DepthMsg::ConstSharedPtr depth_msg,
    typename JointMsg::ConstSharedPtr joint_msg) const {
  cv_bridge::CvImagePtr color_ptr, depth_ptr;
  color_ptr =
      cv_bridge::toCvCopy(color_msg, sensor_msgs::image_encodings::BGR8);
  depth_ptr = cv_bridge::toCvCopy(depth_msg, "16UC1");

  // Rectify the color and depth
  cv::Mat image, depth;
  m_realsense_camera->Rectify(color_ptr->image, depth_ptr->image, image, depth);
  cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

  // Save joint angles
  Eigen::Map<const Eigen::VectorXd> raw_joint_angles(
      joint_msg->position.data(), joint_msg->position.size());
  Eigen::VectorXd joint_angles(m_robot_model.nq);
  joint_angles.segment<4>(0) = raw_joint_angles.segment<4>(0);
  joint_angles.segment<4>(4) = raw_joint_angles.segment<4>(12);
  joint_angles.segment<8>(8) = raw_joint_angles.segment<8>(4);
  m_joint_angles_v.push_back(std::move(joint_angles));

  // Add frame
  FramePtr curr_frame = std::make_shared<Frame>();
  if (m_num_frames == 0) {
    Initialize(image, depth, m_joint_angles_v.back(), curr_frame);
  } else {
    const auto prev_frame = m_frames_v.back();
    Iterate(image, depth, m_joint_angles_v.back(), prev_frame, curr_frame);
  }

  m_frames_v.push_back(curr_frame);
  m_num_frames++;
}

template <typename ColorMsg, typename DepthMsg>
void SAMPublisher<ColorMsg, DepthMsg>::ProjectGraspCenter(
    const std::vector<Eigen::Vector3d> &finger_tips,
    std::vector<Eigen::Vector2d> &finger_tip_centers,
    Eigen::Vector2d &grasp_center) const {
  Eigen::Vector3d point_w = Eigen::Vector3d::Zero();
  for (const auto &finger_tip : finger_tips) {
    point_w += finger_tip;
  }

  point_w /= finger_tips.size();
  point_w[2] += m_sam_offset;

  Eigen::Vector3d point_c =
      m_pose_wc.leftCols<3>().transpose() * (point_w - m_pose_wc.col(3));

  grasp_center = point_c.head<2>() / point_c[2];
  grasp_center[0] = -m_intrinsics[0] * grasp_center[0] + m_intrinsics[2];
  grasp_center[1] = m_intrinsics[1] * grasp_center[1] + m_intrinsics[3];

  for (const auto &tip_w : finger_tips) {
    Eigen::Vector3d tip_c =
        m_pose_wc.leftCols<3>().transpose() * (tip_w - m_pose_wc.col(3));
    Eigen::Vector2d center = tip_c.head<2>() / tip_c[2];
    center[0] = -m_intrinsics[0] * center[0] + m_intrinsics[2];
    center[1] = m_intrinsics[1] * center[1] + m_intrinsics[3];
    finger_tip_centers.push_back(std::move(center));
  }
}

template <typename ColorMsg, typename DepthMsg>
void SAMPublisher<ColorMsg, DepthMsg>::GetFingerTips(
    const Eigen::VectorXd &joint_angles,
    std::vector<Eigen::Vector3d> &finger_tips) const {
  gum::utils::GetFingerTips(m_robot_model, joint_angles, m_base_pose,
                            m_finger_offset, m_finger_ids, finger_tips);
}

template <typename ColorMsg, typename DepthMsg>
void SAMPublisher<ColorMsg, DepthMsg>::ExtractKeyPoints(
    FramePtr curr_frame, const uint8_t *mask_ptr) const {
  // Feature Extraction
  cv::Mat cropped_image;
  const cv::Mat mask(curr_frame->image.size(), CV_8U, (void *)mask_ptr);
  curr_frame
      ->image(cv::Range(curr_frame->bbox[1], curr_frame->bbox[3]),
              cv::Range(curr_frame->bbox[0], curr_frame->bbox[2]))
      .copyTo(cropped_image,
              mask(cv::Range(curr_frame->bbox[1], curr_frame->bbox[3]),
                   cv::Range(curr_frame->bbox[0], curr_frame->bbox[2])));
  cv::cvtColor(cropped_image, cropped_image, cv::COLOR_RGB2GRAY);
  m_superpoint->Extract(cropped_image, m_initial_keypoints_v,
                        m_initial_normalized_keypoints_v,
                        m_initial_keypoint_scores_v, m_initial_descriptors_v);

  for (auto &initial_keypoint : m_initial_keypoints_v) {
    initial_keypoint += curr_frame->offset;
  }
  int num_initial_keypoints = m_initial_keypoints_v.size();
  gum::perception::utils::SelectKeyPointsByDepth(
      num_initial_keypoints, m_min_depth, m_max_depth, m_depth_scale,
      curr_frame->depth, m_initial_keypoints_v, m_initial_descriptors_v,
      m_initial_normalized_keypoints_v, curr_frame->keypoints_v,
      curr_frame->descriptors_v, curr_frame->normalized_keypoints_v);
  gum::perception::utils::GetPointClouds(
      curr_frame->keypoints_v.size(), m_intrinsics[0], m_intrinsics[1],
      m_intrinsics[2], m_intrinsics[3], m_depth_scale, curr_frame->depth,
      curr_frame->keypoints_v, curr_frame->point_clouds_v);
}

template <typename ColorMsg, typename DepthMsg>
void SAMPublisher<ColorMsg, DepthMsg>::RefineKeyPoints(
    FramePtr curr_frame) const {
  auto is_valid = curr_frame->mask_cpu.data_ptr<uint8_t>();
  int num_keypoints = curr_frame->keypoints_v.size();
  m_initial_keypoints_v.clear();
  m_initial_descriptors_v.clear();
  m_initial_normalized_keypoints_v.clear();

  for (int n = 0; n < num_keypoints; n++) {
    const auto &keypoint = curr_frame->keypoints_v[n];
    int x = std::round(keypoint[0]);
    int y = std::round(keypoint[1]);
    if (is_valid[y * m_width + x]) {
      m_initial_keypoints_v.push_back(curr_frame->keypoints_v[n]);
      m_initial_descriptors_v.push_back(curr_frame->descriptors_v[n]);
      m_initial_normalized_keypoints_v.push_back(
          curr_frame->normalized_keypoints_v[n]);
    }
  }
  std::swap(m_initial_keypoints_v, curr_frame->keypoints_v);
  std::swap(m_initial_descriptors_v, curr_frame->descriptors_v);
  std::swap(m_initial_normalized_keypoints_v,
            curr_frame->normalized_keypoints_v);

  gum::perception::utils::GetPointClouds(
      curr_frame->keypoints_v.size(), m_intrinsics[0], m_intrinsics[1],
      m_intrinsics[2], m_intrinsics[3], m_depth_scale, curr_frame->depth,
      curr_frame->keypoints_v, curr_frame->point_clouds_v);
}

template <typename ColorMsg, typename DepthMsg>
void SAMPublisher<ColorMsg, DepthMsg>::WriteFrame(FrameConstPtr frame) const {
  cv::Mat image;
  cv::cvtColor(frame->image, image, CV_RGB2BGR);

  const cv::Mat mask(image.size(), CV_8U, frame->mask_cpu.data_ptr<uint8_t>());
  cv::imwrite(
      m_result_path + "image_" + std::to_string(frame->id) + "_mask.png", mask);

  cv::Mat masked_image;
  image.copyTo(masked_image, mask);
  cv::imwrite(m_result_path + "image_" + std::to_string(frame->id) +
                  "_masked_color.jpg",
              masked_image);

  cv::Mat masked_depth;
  frame->depth.copyTo(masked_depth, mask);
  cv::imwrite(m_result_path + "image_" + std::to_string(frame->id) +
                  "_masked_depth.png",
              masked_depth);

  std::vector<cv::KeyPoint> cv_keypoints_v;
  for (const auto &keypoint : frame->keypoints_v) {
    cv_keypoints_v.push_back({keypoint[0], keypoint[1], 1});
  }
  cv::Mat masked_image_with_keypoints;
  cv::drawKeypoints(masked_image, cv_keypoints_v, masked_image_with_keypoints,
                    cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
  cv::imwrite(m_result_path + "image_" + std::to_string(frame->id) +
                  "_keypoints.jpg",
              masked_image_with_keypoints);

  cv::rectangle(image, cv::Point(frame->bbox[0], frame->bbox[1]),
                cv::Point(frame->bbox[2], frame->bbox[3]),
                cv::Scalar(0, 0, 255), 1, cv::LINE_8);
  cv::imwrite(m_result_path + "image_" + std::to_string(frame->id) +
                  "_bbox.jpg",
              image);
}

template <typename ColorMsg, typename DepthMsg>
void SAMPublisher<ColorMsg, DepthMsg>::WriteMatch(
    FrameConstPtr prev_frame, FrameConstPtr curr_frame,
    const std::vector<Eigen::Vector2i> &matches_v) const {
  std::vector<cv::KeyPoint> prev_cv_keypoints_v;
  std::vector<cv::KeyPoint> curr_cv_keypoints_v;
  std::vector<cv::DMatch> cv_matches_v;

  for (const auto &keypoint : prev_frame->keypoints_v) {
    prev_cv_keypoints_v.push_back({keypoint[0] - prev_frame->offset[0],
                                   keypoint[1] - prev_frame->offset[1], 1});
  }

  for (const auto &keypoint : curr_frame->keypoints_v) {
    curr_cv_keypoints_v.push_back({keypoint[0] - curr_frame->offset[0],
                                   keypoint[1] - curr_frame->offset[1], 1});
  }

  for (const auto &match : matches_v) {
    assert(match[0] >= 0 && match[0] < (int)prev_cv_keypoints_v.size());
    assert(match[1] >= 0 && match[1] < (int)curr_cv_keypoints_v.size());
    cv_matches_v.push_back({match[0], match[1], 1});
  }

  auto prev_cropped_image =
      prev_frame->image(cv::Range(prev_frame->bbox[1], prev_frame->bbox[3]),
                        cv::Range(prev_frame->bbox[0], prev_frame->bbox[2]));
  auto curr_cropped_image =
      curr_frame->image(cv::Range(curr_frame->bbox[1], curr_frame->bbox[3]),
                        cv::Range(curr_frame->bbox[0], curr_frame->bbox[2]));

  cv::Mat cv_match_image;
  std::vector<char> cv_match_mask_v(cv_matches_v.size(), 1);
  cv::drawMatches(prev_cropped_image, prev_cv_keypoints_v, curr_cropped_image,
                  curr_cv_keypoints_v, cv_matches_v, cv_match_image,
                  cv::Scalar::all(-1), cv::Scalar::all(-1), cv_match_mask_v,
                  cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
  cv::imwrite(m_result_path + "image_" + std::to_string(curr_frame->id) +
                  "_matched.jpg",
              cv_match_image);
}

template <typename ColorMsg, typename DepthMsg>
void SAMPublisher<ColorMsg, DepthMsg>::Publish(
    FrameConstPtr frame, const std_msgs::msg::Header &header) const {
  cv::Mat masked_depth;
  frame->depth.copyTo(
      masked_depth,
      cv::Mat(frame->depth.size(), CV_8U, frame->mask_cpu.data_ptr<uint8_t>()));
  sensor_msgs::msg::Image::SharedPtr msg =
      cv_bridge::CvImage(header, "16UC1", masked_depth).toImageMsg();
  msg->header.frame_id = std::to_string(frame->id);
  m_seg_depth_publisher->publish(*msg);
  msg->header.frame_id = frame->id;
}

template <typename ColorMsg, typename DepthMsg>
void SAMPublisher<ColorMsg, DepthMsg>::CallBack(
    typename ColorMsg::ConstSharedPtr color_msg,
    typename DepthMsg::ConstSharedPtr depth_msg,
    typename JointMsg::ConstSharedPtr joint_msg) const {
  RCLCPP_INFO_STREAM(
      this->get_logger(),
      "=========================================================");
  RCLCPP_INFO_STREAM(this->get_logger(), "Frame " << m_frames_v.size());
  RCLCPP_INFO_STREAM(
      this->get_logger(),
      "---------------------------------------------------------");

  AddFrame(color_msg, depth_msg, joint_msg);

  auto curr_frame = m_frames_v.back();
  Publish(curr_frame, depth_msg->header);
}

template class SAMPublisher<sensor_msgs::msg::CompressedImage,
                            sensor_msgs::msg::Image>;
template class SAMPublisher<sensor_msgs::msg::Image, sensor_msgs::msg::Image>;
} // namespace perception
} // namespace gum