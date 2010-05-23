/* Class definition of Ph_Member */

ObjClassField MemberFields[] = {
  { "frame", "unknown" }, /* ignore the type */
  { "index", "int" },
  { "distance", "double" },
  { NULL, NULL }
};

struct ObjClassStruct Member = {
  "member", sizeof(struct MemberData),
  MemberFields, NULL, NULL
};
