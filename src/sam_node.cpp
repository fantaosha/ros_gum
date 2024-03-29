#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include <gum_perception/sam_publisher.h>
#include <rclcpp/rclcpp.hpp>

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);

  rclcpp::spin(std::make_shared<gum::perception::SAMPublisher>("sam"));

  rclcpp::shutdown();
  return 0;
}