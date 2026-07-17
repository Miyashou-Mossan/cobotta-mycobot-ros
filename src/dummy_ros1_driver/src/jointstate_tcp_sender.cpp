#include <ros/ros.h>
#include <sensor_msgs/JointState.h>

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

class JointStateTcpSender
{
public:
    JointStateTcpSender(const std::string& ip, int port)
        : socket_fd_(-1), ip_(ip), port_(port)
    {
    }

    ~JointStateTcpSender()
    {
        if (socket_fd_ >= 0)
        {
            close(socket_fd_);
        }
    }

    bool connectToCobotta()
    {
        socket_fd_ = socket(AF_INET, SOCK_STREAM, 0);

        if (socket_fd_ < 0)
        {
            perror("socket");
            return false;
        }

        sockaddr_in server_addr{};
        server_addr.sin_family = AF_INET;
        server_addr.sin_port = htons(port_);

        if (inet_pton(AF_INET, ip_.c_str(), &server_addr.sin_addr) <= 0)
        {
            ROS_ERROR("Invalid IP address: %s", ip_.c_str());
            return false;
        }

        ROS_INFO("Connecting to COBOTTA at %s:%d", ip_.c_str(), port_);

        if (connect(
                socket_fd_,
                reinterpret_cast<sockaddr*>(&server_addr),
                sizeof(server_addr)) < 0)
        {
            perror("connect");
            return false;
        }

        ROS_INFO("Connected to COBOTTA.");
        return true;
    }

    void jointStateCallback(const sensor_msgs::JointState::ConstPtr& msg)
    {
        if (msg->position.size() < 6)
        {
            ROS_WARN_THROTTLE(
                1.0,
                "JointState has fewer than 6 positions: %zu",
                msg->position.size());
            return;
        }

        std::ostringstream stream;
        stream << std::fixed << std::setprecision(6);

        for (std::size_t i = 0; i < 6; ++i)
        {
            if (i > 0)
            {
                stream << ",";
            }

            stream << msg->position[i];
        }

        const std::string payload = stream.str();
	const std::string message = payload + "\n";

        const ssize_t sent_size = send(
            socket_fd_,
            message.c_str(),
            message.size(),
            0);

        if (sent_size < 0)
        {
            perror("send");
            ros::shutdown();
            return;
        }

        ROS_INFO_THROTTLE(1.0, "Sent: %s", payload.c_str());
    }

private:
    int socket_fd_;
    std::string ip_;
    int port_;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "jointstate_tcp_sender");
    ros::NodeHandle nh;

    JointStateTcpSender sender("192.168.32.68", 5000);

    if (!sender.connectToCobotta())
    {
        ROS_ERROR("Failed to connect to COBOTTA.");
        return 1;
    }

    ros::Subscriber subscriber = nh.subscribe(
        "/joint_states",
        10,
        &JointStateTcpSender::jointStateCallback,
        &sender);

    ROS_INFO("Waiting for /joint_states...");

    ros::spin();

    return 0;
}
