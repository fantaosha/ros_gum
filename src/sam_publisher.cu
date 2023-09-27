#include <cv_bridge/cv_bridge.h>
#include <gum_perception/sam_publisher.h>

#include <gum/perception/feature/outlier_rejection.h>
#include <gum/perception/utils/utils.cuh>
#include <gum/utils/cuda_utils.cuh>
#include <gum/utils/utils.h>

#include <opencv2/imgproc.hpp>

namespace gum {
namespace perception {
SAMPublisher::SAMPublisher(const std::string &node_name)
    : rclcpp::Node(node_name) {
  igraph_rng_seed(igraph_rng_default(), 0);

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

  this->declare_parameter("color_topic", rclcpp::PARAMETER_STRING);
  this->declare_parameter("depth_topic", rclcpp::PARAMETER_STRING);
  this->declare_parameter("segmentation_topic", rclcpp::PARAMETER_STRING);
  this->declare_parameter("model_path", rclcpp::PARAMETER_STRING);
  this->declare_parameter("sam_encoder", rclcpp::PARAMETER_STRING);
  this->declare_parameter("sam_decoder", rclcpp::PARAMETER_STRING);
  this->declare_parameter("superpoint", rclcpp::PARAMETER_STRING);
  this->declare_parameter("lightglue", rclcpp::PARAMETER_STRING);
  this->declare_parameter("ostrack", rclcpp::PARAMETER_STRING);
  this->declare_parameter("trt_engine_cache", rclcpp::PARAMETER_STRING);

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
  const std::string color_topic =
      this->get_parameter("color_topic").as_string();
  const std::string depth_topic =
      this->get_parameter("depth_topic").as_string();
  const std::string sam_topic =
      this->get_parameter("segmentation_topic").as_string();
  const std::string model_path = this->get_parameter("model_path").as_string();
  const std::string sam_encoder_checkpoint =
      model_path + this->get_parameter("sam_encoder").as_string();
  const std::string sam_decoder_checkpoint =
      model_path + this->get_parameter("sam_decoder").as_string();
  const std::string superpoint_checkpoint =
      model_path + this->get_parameter("superpoint").as_string();
  const std::string lightglue_checkpoint =
      model_path + this->get_parameter("lightglue").as_string();
  const std::string ostrack_checkpoint =
      model_path + this->get_parameter("ostrack").as_string();
  const std::string trt_engine_cache_path =
      model_path + this->get_parameter("trt_engine_cache").as_string();

  m_segmentation_publisher =
      this->create_publisher<sensor_msgs::msg::Image>(sam_topic, 10);
  m_color_subscriber = std::make_shared<
      message_filters::Subscriber<sensor_msgs::msg::CompressedImage>>(
      this, color_topic);
  m_depth_subscriber =
      std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(
          this, depth_topic);
  m_synchronizer =
      std::make_shared<message_filters::Synchronizer<ApproximatePolicy>>(
          ApproximatePolicy(10), *m_color_subscriber, *m_depth_subscriber);
  m_synchronizer->registerCallback(&SAMPublisher::CallBack, this);

  CHECK_CUDA(cudaSetDevice(m_device));
  m_handle = std::make_shared<gum::graph::Handle>();

  m_sam = std::make_shared<gum::perception::segmentation::SAM>(
      sam_encoder_checkpoint, sam_decoder_checkpoint, m_device);
  m_superpoint = std::make_shared<gum::perception::feature::SuperPoint>(
      superpoint_checkpoint, trt_engine_cache_path, m_device);
  m_lightglue = std::make_shared<gum::perception::feature::LightGlue>(
      lightglue_checkpoint, trt_engine_cache_path, m_device);
  m_ostracker = std::make_shared<gum::perception::bbox::OSTrack>(
      ostrack_checkpoint, trt_engine_cache_path, m_device);

  m_dataset = std::make_shared<gum::perception::dataset::RealSenseDataset<
      gum::perception::dataset::Device::GPU>>(
      m_device, m_height, m_width, m_intrinsics[0], m_intrinsics[1],
      m_intrinsics[2], m_intrinsics[3], m_intrinsics[4], m_intrinsics[5],
      m_intrinsics[6], m_intrinsics[7], m_depth_scale);

  RCLCPP_INFO_STREAM_ONCE(this->get_logger(), "Frontend Publisher Setup");
}

void SAMPublisher::Process(const Frame &prev_frame, Frame &curr_frame) {
  std::vector<Eigen::Vector2f> initial_keypoints_v;
  std::vector<float> initial_keypoint_scores_v;
  std::vector<Eigen::Vector<float, 256>> initial_descriptors_v;
  std::vector<Eigen::Vector2f> initial_normalized_keypoints_v;

  const Eigen::Vector4f &init_bbox = prev_frame.bbox.cast<float>();

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

  cv::Mat initial_cropped_image;
  curr_frame
      .image(cv::Range(curr_frame.bbox[1], curr_frame.bbox[3]),
             cv::Range(curr_frame.bbox[0], curr_frame.bbox[2]))
      .copyTo(initial_cropped_image,
              cv::Mat(curr_frame.image.size(), CV_8U,
                      extended_mask_cpu.data_ptr<uint8_t>())(
                  cv::Range(curr_frame.bbox[1], curr_frame.bbox[3]),
                  cv::Range(curr_frame.bbox[0], curr_frame.bbox[2])));

  // Initial Mask
  cv::cvtColor(initial_cropped_image, initial_cropped_image,
               cv::COLOR_RGB2GRAY);

  // Extract Keypoints
  m_superpoint->Extract(initial_cropped_image, initial_keypoints_v,
                        initial_normalized_keypoints_v,
                        initial_keypoint_scores_v, initial_descriptors_v);
  for (auto &initial_keypoint : initial_keypoints_v) {
    initial_keypoint += curr_frame.offset;
  }

  int num_initial_keypoints = initial_keypoints_v.size();
  gum::perception::utils::SelectKeyPointsByDepth(
      num_initial_keypoints, m_min_depth, m_max_depth, m_depth_scale,
      curr_frame.depth, initial_keypoints_v, initial_descriptors_v,
      initial_normalized_keypoints_v, curr_frame.keypoints_v,
      curr_frame.descriptors_v, curr_frame.normalized_keypoints_v);

  // Point Clouds
  int num_keypoints = curr_frame.keypoints_v.size();
  gum::perception::utils::GetPointClouds(
      num_keypoints, m_intrinsics[0], m_intrinsics[1], m_intrinsics[2],
      m_intrinsics[3], m_depth_scale, curr_frame.depth, curr_frame.keypoints_v,
      curr_frame.point_clouds_v);

  // Feature Matching
  thrust::device_vector<int> d_initial_matches_v;
  std::vector<float> match_scores_v;
  std::vector<Eigen::Vector2i> initial_matches_v;
  m_lightglue->Match(prev_frame.normalized_keypoints_v,
                     curr_frame.normalized_keypoints_v,
                     prev_frame.descriptors_v, curr_frame.descriptors_v,
                     initial_matches_v, match_scores_v);
  int num_initial_matches = initial_matches_v.size();
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
  m_sam->SetImage(curr_frame.image);
  m_sam->Query(point_coords_v, point_labels_v, init_bbox, masks, scores,
               logits);
  curr_frame.mask_gpu = masks[0][1].to(torch::kUInt8);
  curr_frame.mask_cpu = curr_frame.mask_gpu.to(torch::kCPU);
  gum::perception::utils::FilterMaskByDepth(
      m_height, m_width, curr_frame.bbox, m_min_depth, m_max_depth,
      m_depth_scale, curr_frame.depth, curr_frame.mask_cpu.data_ptr<uint8_t>());
  gum::perception::utils::RefineMask(m_height, m_width, curr_frame.bbox,
                                     curr_frame.mask_cpu.data_ptr<uint8_t>());
  curr_frame.mask_gpu = curr_frame.mask_cpu.to(curr_frame.mask_gpu.device());
}

void SAMPublisher::CallBack(
    const sensor_msgs::msg::CompressedImage::ConstSharedPtr &color_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr &depth_msg) {
  cv_bridge::CvImagePtr color_ptr, depth_ptr;
  color_ptr =
      cv_bridge::toCvCopy(color_msg, sensor_msgs::image_encodings::BGR8);
  depth_ptr = cv_bridge::toCvCopy(depth_msg, "16UC1");
  double timestamp = double(color_msg->header.stamp.sec) +
                     1e-9 * double(color_msg->header.stamp.nanosec);
  m_dataset->AddFrame(
      {timestamp, std::move(color_ptr->image), std::move(depth_ptr->image)});

  //   cv::imwrite(std::string(color_ptr->header.frame_id) + ".jpg",
  //               m_dataset->GetFrames().back().image);
  if (m_dataset->GetNumFrames() >= 1000) {
    m_dataset->Clear();
  }
}
} // namespace perception
} // namespace gum