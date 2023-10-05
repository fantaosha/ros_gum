#include <memory.h>
#include <signal.h>
#include <stdio.h>
#include <termios.h>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/char.hpp>

#define STOP 'w'
#define START 's'
#define PAUSE 'a'
#define RESUME 'd'

namespace gum {
namespace perception {
class SAMCommandNode : public rclcpp::Node {
public:
  SAMCommandNode(std::string node_name) : rclcpp::Node(node_name) {
    this->declare_parameter("command_topic", rclcpp::PARAMETER_STRING);
    m_cmd_topic = this->get_parameter("command_topic").as_string();
    m_cmd_publisher =
        this->create_publisher<std_msgs::msg::Char>(m_cmd_topic, 20);
  }

  void Loop() {
    RCLCPP_INFO_STREAM(this->get_logger(), "---------------------------");
    RCLCPP_INFO_STREAM(this->get_logger(), "Reading from keyboard");
    RCLCPP_INFO_STREAM(this->get_logger(), "---------------------------");
    RCLCPP_INFO_STREAM(this->get_logger(), "'w': STOP");
    RCLCPP_INFO_STREAM(this->get_logger(), "'s': START");
    RCLCPP_INFO_STREAM(this->get_logger(), "'a': PAUSE");
    RCLCPP_INFO_STREAM(this->get_logger(), "'d': RESUME");

    std::string cmd_info;
    std_msgs::msg::Char cmd;
    while (rclcpp::ok()) {
      char c = getch();
      bool dirty = false;

      switch (c) {
      case STOP:
        cmd_info = "STOP";
        dirty = true;
        break;
      case START:
        cmd_info = "START";
        dirty = true;
        break;
      case PAUSE:
        cmd_info = "PAUSE";
        dirty = true;
        break;
      case RESUME:
        cmd_info = "RESUME";
        dirty = true;
        break;
      }

      if (dirty == true) {
        RCLCPP_INFO_STREAM(this->get_logger(),
                           "FRONTEND COMMAND: " << cmd_info);
        cmd.data = c;
        m_cmd_publisher->publish(cmd);
      }
    }
  }

private:
  rclcpp::Publisher<std_msgs::msg::Char>::SharedPtr m_cmd_publisher;
  std::string m_cmd_topic;

  char getch() {
    fd_set set;
    struct timeval timeout;
    int rv;
    char buff = 0;
    int len = 1;
    int filedesc = 0;
    FD_ZERO(&set);
    FD_SET(filedesc, &set);

    timeout.tv_sec = 0;
    timeout.tv_usec = 1000;

    rv = select(filedesc + 1, &set, NULL, NULL, &timeout);

    struct termios old = {0};
    if (tcgetattr(filedesc, &old) < 0)
      RCLCPP_ERROR_STREAM(this->get_logger(), "tcsetattr()");
    old.c_lflag &= ~ICANON;
    old.c_lflag &= ~ECHO;
    old.c_cc[VMIN] = 1;
    old.c_cc[VTIME] = 0;
    if (tcsetattr(filedesc, TCSANOW, &old) < 0)
      RCLCPP_ERROR_STREAM(this->get_logger(), "tcsetattr ICANON");

    if (rv == -1)
      RCLCPP_ERROR_STREAM(this->get_logger(), "select");
    else
      read(filedesc, &buff, len);

    old.c_lflag |= ICANON;
    old.c_lflag |= ECHO;
    if (tcsetattr(filedesc, TCSADRAIN, &old) < 0)
      RCLCPP_ERROR_STREAM(this->get_logger(), "tcsetattr ~ICANON");
    return (buff);
  }
};
} // namespace perception
} // namespace gum

int main(int argc, char **argv) {
  //   ros::init(argc, argv, "teleop_turtle");
  rclcpp::init(argc, argv);

  auto sam_commander =
      std::make_shared<gum::perception::SAMCommandNode>("sam_cmd");
  sam_commander->Loop();

  rclcpp::shutdown();
  return 0;
}