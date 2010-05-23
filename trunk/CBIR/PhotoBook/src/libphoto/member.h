/* Definition of the Photobook database member "Ph_Member" structure */

struct MemberData {
  Frame frame;
  int index;
  double distance;
};

typedef Ph_Object Ph_Member;

#define Ph_MemFrame(m) ((struct MemberData *)((m)->data))->frame
#define Ph_MemName(m) frame_name(Ph_MemFrame(m))
#define Ph_MemIndex(m) ((struct MemberData *)((m)->data))->index
#define Ph_MemDistance(m) ((struct MemberData *)((m)->data))->distance
