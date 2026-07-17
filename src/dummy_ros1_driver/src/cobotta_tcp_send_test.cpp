#include <iostream>
#include <cstring>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>

int main()
{
    const char* cobotta_ip = "192.168.32.68";
    const int port = 5000;

    int sockfd = socket(AF_INET, SOCK_STREAM, 0);

    if (sockfd < 0)
    {
        perror("socket");
        return 1;
    }

    sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);

    if (inet_pton(AF_INET, cobotta_ip, &server_addr.sin_addr) <= 0)
    {
        std::cerr << "Invalid IP address" << std::endl;
        close(sockfd);
        return 1;
    }

    std::cout << "Connecting to "
              << cobotta_ip << ":" << port << "..." << std::endl;

    if (connect(
            sockfd,
            reinterpret_cast<sockaddr*>(&server_addr),
            sizeof(server_addr)) < 0)
    {
        perror("connect");
        close(sockfd);
        return 1;
    }

    std::cout << "Connected." << std::endl;

    const char message[] = "0.1,0.2,0.3,0.4,0.5,0.6";

    ssize_t sent_size = send(
        sockfd,
        message,
        std::strlen(message),
        0);

    if (sent_size < 0)
    {
        perror("send");
        close(sockfd);
        return 1;
    }

    std::cout << "Sent: " << message << std::endl;

    close(sockfd);

    return 0;
}
