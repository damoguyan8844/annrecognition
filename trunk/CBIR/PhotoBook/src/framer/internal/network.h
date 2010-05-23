/* C */

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <stdio.h>
#include <fcntl.h>
#include <errno.h>

/* This file implements declarations and macros for dealing with published
   operations and network operations in general. */

#define MAX_CONNECTIONS 30

enum PARSE_STATE { in_string, in_escape, in_string_escape, in_comment, other };
struct FCONNECTION {FILE *in_stream; int socket; 
		    char *hostname; struct in_addr host_addr[4];
		    enum PARSE_STATE state; int parse_depth;
		    struct STRING_OUTPUT_STREAM s_out;};
extern struct FCONNECTION connections[];
extern int number_of_connections;

extern boolean isolationism;
void publish(char *op,Frame topic,Grounding extra);
void distribute(char *op,Frame topic,Grounding extra);
struct hostent *get_local_host();
struct servent *get_tcp_service(char *service);
struct FCONNECTION *get_server_fn(Frame f);

#define BEGIN_PUBLISHED_OP(name,topic,extra) \
   if ((NOT(isolationism)) && (remote_frame_p(topic))) \
     publish(name,topic,extra); \
   {FLET(boolean,isolationism,True)

#define END_PUBLISHED_OP(name,topic,extra) \
   END_FLET(isolationism);} \
   if ((NOT(isolationism)) && (server_p) && (NOT(remote_frame_p(topic)))) \
     distribute(name,topic,extra); 

#define get_server(f) \
  (((frame_type(frame)) == remote_frame) ? (get_server_fn(f)) : \
   (connections[frame_type(frame)-16]))
   
