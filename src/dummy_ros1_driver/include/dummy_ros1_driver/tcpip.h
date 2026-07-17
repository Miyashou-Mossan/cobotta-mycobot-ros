#ifndef _TCPIP_H
#define _TCPIP_H
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <iostream>
#include <arpa/inet.h>

class TcpIp
{
 public:
 int server_socket(unsigned short portnum);
 int client_socket(char* address, unsigned short portnum);
 int sock_sendc(char* buf);
 int sock_recvc(char*buf);
 int sock_send(char* buf);
 int sock_recv(char* buf);
 int sock_close();
 private :
 char* ip_address;
 unsigned short port;
 int sockfd;
 int client_sockfd;
 struct sockaddr_in addr;
 struct sockaddr_in from_addr;
 char msg_r[50];
 

 
};
#endif
