#include "dummy_ros1_driver/tcpip.h"
int TcpIp::server_socket(unsigned short portnum){
  port = portnum;
  socklen_t len = sizeof( struct sockaddr_in );
  
  if((sockfd = socket( AF_INET, SOCK_STREAM, 0 ) ) < 0 ) {
    perror( "socket" );
  }
  
  addr.sin_family = AF_INET;
  addr.sin_port = htons( port  );
  addr.sin_addr.s_addr = INADDR_ANY;
  if( bind( sockfd, (struct sockaddr *)&addr, sizeof( addr ) ) < 0 ) {
    perror( "bind" );
  }
  if( listen( sockfd, SOMAXCONN ) < 0 ) {
    perror( "listen" );
  }
  
  
  if( ( client_sockfd = accept( sockfd, (struct sockaddr *)&from_addr, &len ) ) < 0) {
    perror( "accept" );
  }
  return 0;
}   
int TcpIp::client_socket(char* address, unsigned short portnum){
  ip_address = address;
  port = portnum;
  if( (sockfd = socket(AF_INET, SOCK_STREAM, 0) ) < 0) {
    perror( "socket" );
  }
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  addr.sin_addr.s_addr = inet_addr(ip_address);
  connect( sockfd, (struct sockaddr *)&addr, sizeof( struct sockaddr_in ) );
  return 0;
}

int TcpIp::sock_send(char* buf){
  write(sockfd,buf,50);
  std::cout<<"Send:"<<buf<<std::endl;
  return 0;
}

int TcpIp::sock_recv(char* buf){
  read(sockfd, buf,50);
  std::cout<<"Recv:"<<buf<<std::endl;
  return 0;
}

int TcpIp::sock_recvc(char* buf){
  memset(msg_r,' ',50);
  read(client_sockfd,msg_r,50);
  std::cout<<"Recv:"<<msg_r<<std::endl;
  return 0;
}

int TcpIp::sock_sendc(char* buf){
    write(client_sockfd,buf,50);
    std::cout<<"Send:"<<buf<<std::endl;
    return 0;
}

int TcpIp::sock_close(){
 close(sockfd);
 std::cout<<"Socket close"<<std::endl;
 return 0;
}
