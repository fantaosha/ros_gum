cmake_minimum_required(VERSION 3.8)
project(gum_perception)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

include_directories(./include)

find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(cv_bridge REQUIRED)

find_package(GUM REQUIRED)
find_package(igraph REQUIRED)
find_package(Torch REQUIRED)

add_executable(sam_publisher include/gum_perception/sam_publisher.h src/sam_node.cpp src/sam_publisher.cu)
target_link_libraries(sam_publisher gum ${TORCH_LIBRARIES})
ament_target_dependencies(sam_publisher rclcpp std_msgs message_filters sensor_msgs cv_bridge)

install(TARGETS sam_publisher DESTINATION lib/${PROJECT_NAME})
