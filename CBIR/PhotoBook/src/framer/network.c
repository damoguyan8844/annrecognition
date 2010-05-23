/* C */

/* This is code for a FRAMER (FRAXL) server program */

#include "framer.h"
#include "fraxl.h"
#include "string.h"
#if (NETWORKING)
#include <sys/time.h>
#include "internal/network.h"


#define MAXHOSTNAME 100
#define MAX_BACKLOG 50
#define FRAMER_SERVICE "framer"
#define INITIAL_CONNECTION_BUFFER_SIZE 256
#define OPEN_PAREN '('
#define CLOSE_PAREN ')'
#define OPEN_BRACE '{'
#define CLOSE_BRACE '}'

/* This is for interpreting server names using a local map */
Frame name_map; 
/* These keep track of connections in various ways. */
struct FCONNECTION connections[MAX_CONNECTIONS]; int number_of_connections=0;
fd_set all_connections;
/* Setting this to true will cause the server to exit */
boolean exit_server=False;

void xperror(char *info);
exception 
  Not_A_Remote_Frame="Cannot get server for a non remote frame",
  Cant_Find_Server="Cannot find server for frame",
  Not_A_Valid_Server="Not a valid server name";

void check_connections(void);

/* Setting up a new connection */

int make_new_connection(int client_socket,char *hostname)
{
  connections[number_of_connections].socket=client_socket;
  connections[number_of_connections].in_stream=fdopen(client_socket,"r");
  connections[number_of_connections].state=other; 
  connections[number_of_connections].parse_depth=0;
  /* Initialize the string output stream */
  connections[number_of_connections].s_out.original=NULL;
  ALLOCATE(connections[number_of_connections].s_out.head,char,
	   INITIAL_CONNECTION_BUFFER_SIZE);
  connections[number_of_connections].s_out.point=
    connections[number_of_connections].s_out.head;
  connections[number_of_connections].s_out.tail=
    connections[number_of_connections].s_out.head+INITIAL_CONNECTION_BUFFER_SIZE;
  connections[number_of_connections].s_out.head[0]='\0';
  connections[number_of_connections].hostname=hostname;
  fprintf(stderr,";; New connection with %s....[%d]\n",
	  connections[number_of_connections].hostname,number_of_connections);
  /* Add the client socket to the framer clients. */
  FD_SET(client_socket,&all_connections);
  return number_of_connections++;
}


/* Sending expressions out on connections */

void send_expr(struct FCONNECTION *connection,Grounding expr)
{
  char buffer[INITIAL_CONNECTION_BUFFER_SIZE];
  {WITH_OUTPUT_TO_EXISTING_STRING(gs,buffer,INITIAL_CONNECTION_BUFFER_SIZE)
     {print_ground(gs,expr); gsputc('\n',gs);
      send(connection->socket,string_so_far(gs),string_size_so_far(gs),0);}
   END_WITH_OUTPUT_TO_STRING(gs);}
}

/* Serving connections */

void eval_on_connection(struct FCONNECTION *c,char *string)
{
  Grounding nd_apply(Grounding rail);
  Grounding input=NULL, result=NULL;
  WITH_HANDLING
    {input=parse_ground_from_string(string); USE_GROUND(input);
     fprintf(stderr,"; Read from %s: %s\n",c->hostname,string); 
     result=nd_apply(input);
     {fprintf(stderr,"; Application yields: "); 
      fprint_ground(stderr,result);
      fprintf(stderr,"\n"); }
     send_expr(c,result);}
  ON_EXCEPTION
    {send(c->socket,"#;Exception",strlen("#;Exception"),0);
     CLEAR_EXCEPTION();}
  END_HANDLING
    FREE_GROUND(input); FREE_GROUND(result);
}

void serve_connection(struct FCONNECTION *c)
{
  FILE *stream; struct STRING_OUTPUT_STREAM *sstream;
  int ch; enum PARSE_STATE state; int depth; 
  stream=c->in_stream; sstream=(&(c->s_out)); 
  state=c->state; depth=c->parse_depth;
  while ((ch=fgetc(stream)) != EOF)
    {sputc(ch,sstream);
     switch (state)
       {case in_string:
	  if (ch == '\\') state=in_string_escape; break;
	case in_string_escape: state=in_string; break;
	case in_escape: state=other; break;
	case in_comment: if (ch == '\n') state=other; break;
	case other:
	  if ((ch == OPEN_PAREN) || (ch == OPEN_BRACE)) depth++;
          else if ((ch == CLOSE_PAREN) || (ch == CLOSE_BRACE)) 
	    if ((--depth) == 0) 
	      {sstream->point=sstream->head;
	       eval_on_connection(c,sstream->head);}}}
  c->state=state; c->parse_depth=depth;
  return;
}


/* Starting servers */

int open_server_socket(char *service)
{
  struct hostent *local_host; struct servent *service_entry; int socket_id;
  struct sockaddr_in *server_address;
  local_host=get_local_host(); service_entry=get_tcp_service(service);
  ALLOCATE(server_address,struct sockaddr_in,1); 
  memset(server_address,0,sizeof(struct sockaddr_in));
  server_address->sin_port=service_entry->s_port;
  memmove((char *) &((*server_address).sin_addr),(char *) local_host->h_addr,
	local_host->h_length);
  server_address->sin_family=local_host->h_addrtype;
  socket_id=socket(local_host->h_addrtype,SOCK_STREAM,0);
  if (socket_id < 0) 
    raise_crisis_with_details("Can't open socket",service);
  else if ((bind(socket_id,server_address,sizeof(struct sockaddr_in))) < 0)
    raise_crisis_with_details("Can't bind server socket",service);
  else {listen(socket_id,MAX_BACKLOG); fcntl(socket_id,F_SETFL,O_NDELAY);
	return socket_id;}
}

void start_server(char *service)
{
  char *get_hostname_from_address(struct in_addr *addr);
  fd_set listen_set; int server_socket;
  fprintf(stderr,"Starting up FRAMER server..");
  server_socket=open_server_socket(service);
  number_of_connections=server_socket;
  FD_ZERO(&all_connections); FD_SET(server_socket,&all_connections);
  fprintf(stderr,"..(listening).."); 
  {UNWIND_PROTECT
     while (NOT(exit_server))
       {listen_set=all_connections;
	select(number_of_connections+1,&listen_set,NULL,NULL,NULL);
	if (FD_ISSET(server_socket,&listen_set))
	  {struct sockaddr_in client_addr; int client_addr_length, client_socket;
	   client_addr_length=sizeof(client_addr);
	   client_socket=accept(server_socket,(struct sockaddr *) &client_addr,
				&client_addr_length);
	   if (client_socket < 0) 
	     if (errno == EWOULDBLOCK) errno=0; 
	     else xperror("accept");
	   else make_new_connection
	     (client_socket,get_hostname_from_address(&(client_addr.sin_addr)));}
	{DO_TIMES(i,number_of_connections)
	   if (FD_ISSET(connections[i].socket,&listen_set))
	     serve_connection(&(connections[i]));}}
   ON_UNWIND
     close(server_socket);
   END_UNWIND}
}


/* Network utility functions */

static struct servent fake_service_entry;

struct servent *get_tcp_service(char *service)
{
  struct servent *service_entry; char *scan; 
  scan=service; while ((*scan != '\0') && (isdigit(*scan))) scan++;
  if (*scan == '\0')
    {int number; number=strtol(service,NULL,10); 
     fake_service_entry.s_port=htonl(number);
     return &fake_service_entry;}
  else if ((service_entry=getservbyname(service,"tcp")) == NULL)
    {raise_crisis_with_details("Can't resolve service name",service);
     return NULL;}
  else return service_entry;
}

struct hostent *get_local_host()
{
  char local_hostname[MAXHOSTNAME+1];
  struct hostent *local_host; 
  /* Get local host entry */
  gethostname(local_hostname,MAXHOSTNAME);
  if ((local_host=gethostbyname(local_hostname)) == NULL)
    {raise_crisis("Cannot resolve local host name");
     return NULL;}
  else return local_host;
}

char *get_hostname_from_address(struct in_addr *addr)
{
  char *hostname; struct hostent *host;
  host=gethostbyaddr(addr,4,AF_INET);
  ALLOCATE(hostname,char,strlen(host->h_name)+1); 
  strcpy(hostname,host->h_name);
  return hostname;
}


/* Frames to connections */

int open_client_socket(char *tag,char *host,char *service)
{
  struct hostent *remote_host; struct servent *service_entry; int socket_id;
  struct sockaddr_in server_address;
  remote_host=gethostbyname(host); service_entry=get_tcp_service(service);
  socket_id=socket(AF_INET,SOCK_STREAM,0);
  if (socket_id < 0) xperror("socket (to me)");
  server_address.sin_port=service_entry->s_port;
  memmove((char *) &((server_address).sin_addr),(char *) remote_host->h_addr,
	  remote_host->h_length);
  server_address.sin_family=remote_host->h_addrtype;
  if ((connect(socket_id,&server_address,sizeof(struct sockaddr_in))) < 0)
    xperror("connect");
  return make_new_connection(socket_id,tag);
}

Frame_Type get_server_by_name(char *server_name)
{
  char *copy, *atsign, host_name[100], service_name[100];
  {DO_TIMES(i,number_of_connections)
     if (string_compare(connections[i].hostname,server_name) == 0)
       return i;}
  ALLOCATE(copy,char,strlen(server_name)+1); strcpy(copy,server_name);
  atsign=strchr(copy,'@');
  if (NOT(NULLP(atsign)))
    {*atsign='\0'; 
     strcpy(host_name,atsign+1); strcpy(service_name,copy);
     *atsign='@';
     return open_client_socket(copy,host_name,service_name);}
  else return open_client_socket(copy,copy,"framer");
}

struct FCONNECTION *get_server_fn(Frame f)
{
  Frame raw_local_probe_annotation(Frame f,char *name);
  if ((frame_type(f)) == remote_frame)
    {DO_HOMES(provider,f)
       if ((frame_type(provider)) > remote_frame)
	 {set_frame_type(f,frame_type(provider));
	  return &(connections[((int) frame_type(f))-16]);}
       else {Frame server_annotation; int server_id;
	     server_annotation=raw_local_probe_annotation(f,"+server");
	     if (NOT(NULLP(frame_ground(server_annotation))))
	       if (NOT(TYPEP(frame_ground(server_annotation),string_ground)))
		 raise_crisis(Not_A_Valid_Server);
	       else {server_id=
		       (Frame_Type)
			 16+get_server_by_name(GSTRING(frame_ground(server_annotation)));
		     set_frame_type(provider,server_id); set_frame_type(f,server_id);
		     return &(connections[((int) server_id)-16]);}}
       raise_crisis(Cant_Find_Server); return NULL;}
  else if (remote_frame_p(f))
    return &(connections[((int) frame_type(f))-16]);
  else {raise_crisis(Not_A_Remote_Frame); return NULL;}
}


/* FRAXL primitives */

Grounding server(Grounding service_name)
{
  start_server(GSTRING(service_name));
  return NULL;
}

Grounding remote_apply(Grounding host,Grounding expr)
{
  int i; i=(int) get_server_by_name(GSTRING(host));
  send_expr(&(connections[i]),expr);
  return fparse_ground(connections[i].in_stream);
}

Grounding remote_eval(Grounding host,Grounding expr)
{
  Grounding find_function(char *name);
  int i; i=(int) get_server_by_name(GSTRING(host));
  expr=cons_pair(find_function("EVAL"),cons_pair(expr,empty_list));
  USE_GROUND(expr); send_expr(&(connections[i]),expr); FREE_GROUND(expr);
  return fparse_ground(connections[i].in_stream);
}


/* Declarations */

void init_network_functions()
{
  name_map=use_annotation(use_annotation(root_frame,"system"),"netaliases");
  declare_unary_function(server,"SERVER",string_ground);
  declare_binary_function(remote_apply,"REMOTE-APPLY",string_ground,any_ground);
  declare_binary_function(remote_eval,"REVAL",string_ground,any_ground);
}

/* We declare this so that we can trap it */
void xperror(char *info)
{
  perror(info); exit(1);
}

#endif /* (NETWORKING) */
