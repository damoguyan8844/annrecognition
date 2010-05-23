/* Object class definition for RankCombo */

ObjClassField RankComboFields[] = {
  { "num-metrics", "int" },
  { "metrics", "ptr array[?x] quark" },
  { "weights", "quark" },
  { NULL, NULL }
};

/* Implemented in rank_combo.c */
extern GenericFunc RankComboCon, RankComboDes, RankComboDistance;

ObjClassFunc RankComboFuncs[] = {
  { "distance", RankComboDistance },
  { "constructor", RankComboCon },
  { "destructor", RankComboDes },
  { NULL, NULL }
};

struct ObjClassStruct RankCombo = {
  "rank_combo", sizeof(struct RankComboData), 
  RankComboFields, RankComboFuncs, NULL
};

