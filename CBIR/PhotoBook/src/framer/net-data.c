/* Functions for data sharing */

int served_bit=0;

Grounding frame_summary(Frame frame)
{
  Grounding result=empty_list; 
  if (served_bit == 0) served_bit=grab_search_bit();
  set_search_bit(frame,served_bit);
  return generate_frame_summary(frame);
}

Frame update_remote_frame(Frame frame)
{
  Grounding request, summary; struct FCONNECTION *server;
  server=get_server(frame); 
  request=cons_pair(find_function("FRAME-SUMMARY"),
		    cons_pair(frame_to_ground(frame),empty_list));
  send_expr(server->socket,request);
  {FLET(boolean,suppress_autoload,True)
     {Grounding temp, proto, ground; 
      temp=summary=fparse_ground(server->in_stream);
      proto=GCAR(temp); temp=GCDR(temp); ground=GCAR(temp); temp=GCDR(temp);
      if (proto == empty_list) 
	internal_modify_prototype
	  (frame,use_annotation(frame_prototype(home),frame_name(frame)));
      else internal_modify_prototype(frame,GFRAME(proto));
      FREE_GROUND(frame->ground); frame->ground=ground; USE_GROUND(ground);
      {DO_LIST(a_name,temp) use_annotation(frame,GSTRING(a_name));}}
   END_FLET(suppress_autoload);}
  FREE_GROUND(request); FREE_GROUND(summary);
  set_frame_current_p(frame);
  return frame;
}

Frame set_remote_ground(Frame f,Grounding g)
{
  apply2_on_connection
    (get_server(f),find_function("SERVER-SET-GROUND"),
     frame_to_ground(f),g);
  return update_remote_frame(f);
}

Frame set_remote_prototype(Frame f,Frame p)
{
  apply2_on_connection
    (get_server(f),find_function("SERVER-SET-PROTOTYPE"),
     frame_to_ground(f),frame_to_ground(p));
  return update_remote_frame(f);
}

void add_remote_annotation(Frame f,char *name)
{
  apply2_on_connection
    (get_server(f),find_function("SERVER-ADD-ANNOTATION"),
     frame_to_ground(f),string_to_ground(name));
  update_remote_frame(f);
}

Grounding use_server(Frame f,Grounding server_name)
{
  set_ground(use_annotation(f,"+server"),server_name);
  set_frame_type(f,remote_frame);
  return f;
}


/* Server functions */

void invalidate_clients(f)
{
  {UNWIND_PROTECT
     {Frame old_read_root; Frame_Array *old_zip_codes; boolean old_ss;
      old_read_root=read_root; read_root=NULL;
      old_zip_codes=zip_codes; zip_codes=NULL;
      old_ss=strict_syntax; strict_syntax=True;
      {OLD_WITH_OUTPUT_TO_STRING(gs,ss,buffer,128)
	 {gsputc('I',gs); print_frame(gs,f);
	  {DO_TIMES(i,number_of_connections)
	     send(connections[i].socket,ss.head,ss.point-ss.head+1,MSG_OOB);}}
	 END_WITH_OUTPUT_TO_STRING(gs);}}
   ON_UNWIND
     {read_root=old_read_root; zip_codes=old_zip_codes; strict_syntax=old_ss;}
   END_UNWIND}
}

/* Client side of invalidation */
void check_connections()
{
  struct timeval to; fd_set listen_set; boolean any_action=False;
  listen_set=all_connections; to.sec=0; to.usec=100;
  if (select(number_of_connections+1,NULL,NULL,&listen_set,&to))
    {DO_TIMES(i,number_of_connections)
       if (FD_ISSET(connections[i].socket,&listen_set))
	 {char buffer[256]; Grounding message, result; any_action=True;
	  recv(connections[i].socket,buffer,256,MSG_OOB);
	  if (buffer[0] == 'I')
	    {Frame inv; inv=parse_frame_from_string(buf+1);
	     clear_frame_current_p(inv);}}
     check_connections();}
}

Grounding server_set_ground(Frame f,Grounding g)
{
  set_ground(f,g); 
  if (search_bit_p(f,served_bit)) invalidate_clients(f);
  return NULL;
}

Grounding server_set_prototype(Frame f,Frame p)
{
  set_prototype(f,p); 
  if (search_bit_p(f,served_bit)) invalidate_clients(f);
  return NULL;
}

Grounding server_add_annotation(Frame f,Grounding name)
{
  use_annotation(f,GSTRING(name));
  if (search_bit_p(f,served_bit)) invalidate_clients(f);
  return NULL;
}

