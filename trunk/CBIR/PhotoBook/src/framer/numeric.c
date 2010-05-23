/* -*- Mode: C -*- 

  Copyright (c) 1991, 1992 by the Massachusetts Institute of Technology.
  All rights reserved.

  This file is part of FRAMER, a representation language and semantic
  database developed by Kenneth B. Haase and his students at the Media
  Laboratory at the Massachusetts Institute of Technology in Cambridge,
  Massachusetts.  Research at the Media Lab is supported by funds and
  equipment from a variety of corporations and government sponsors whose
  contributions are gratefully acknowledged.

  Permission to use, copy, or modify these programs and their
  documentation for educational and research purposes only and without
  fee is hereby granted, provided that this copyright and permission
  notice appear on all copies and supporting documentation.  For any
  other uses of this software, in original or modified form, including
  but not limited to distribution in whole or in part, specific prior
  permission from M.I.T. must be obtained.  M.I.T. makes no
  representations about the suitability of this software for any
  purpose.  It is provided "as is" without express or implied warranty.

*************************************************************************/

/*Should I replace some of the copying with a pointer count?*/
/*every bignum_to_ground is followed by FREE_BIGNUM...but it's not
  logical to do that in the func, could be a needed param*/

/************************************************************************
  This file defines the core functions for parsing, printing, and reclaiming
  FRAMER grounds.

  Modification history:
    13 June 1992 - Began tracking modification history (Haase)
*************************************************************************/
 
/* This contains the C implementations of various FRAXL primitives. */

#include <limits.h>
#include <errno.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include "framer.h"
#include "fraxl.h"
#include "internal/bignums.h"

/* for debugging: */
#include <assert.h>

int z=0;

/* List stuff....
typedef struct BIGNUM_LIST
{
  BignumType bignum;
  int initline;
  int ptrcount;
  struct BIGNUM_LIST *next;
} bignum_list;

bignum_list *biglist = NULL;
*/

#define debug(x) x

void free_bignum(BignumType num)
{
#ifdef listhuh
  bignum_list *runner = biglist;
  if (biglist->bignum == num) biglist = biglist->next; 
  else
    {
      while (runner->next->bignum != num)  runner=runner->next;
      if (runner->next->ptrcount == 1) {
	free(runner->next);
	runner->next = runner->next->next;
      }
      else runner->next->ptrcount--;
    }
#endif
  free(num);
  z--; 
}
    
#define FREE_BIGNUM(num) free_bignum(num)

#if 0
int noDuplicates(BignumType bn, bignum_list *listptr)
/*returns true if bignum is not in biglist after (not including) listptr */
{
  int nodupes = 1;
  while (listptr->next) {
    listptr = listptr->next;
    if (listptr->bignum == bn) {
      nodupes = 0;
      break;
    }
  }
  return nodupes;
}
#endif

    

extern exception Type_Error;
exception Func_Error = "Math Implementation Error",
          Division_By_Zero = "Division by zero";

static Grounding a_one, a_neg_one, a_zero;



#define debug(x) x


static BignumType initializeBignum(int linenum);
static BignumType grow_bignum(BignumType bignum, int by);
static BignumType addIntToBignum(BignumType bignum, int addend);
static BignumType multBignumByInt(BignumType bignum, int factor);


/* Generic math routines */
Grounding generic_plus(Grounding x, Grounding y)
{
  switch (ground_type(x))
    {
    case (integer_ground):
      switch (ground_type(y))
	{
	case integer_ground: {
	  double result = (double) GINTEGER(x) + GINTEGER(y);
	  if ((result > INT_MAX) || (result < INT_MIN))
	    {
	      Grounding result;
	      BignumType temp;
	      BignumType bigx = signedIntToBignum(GINTEGER(x));
	      BignumType bigy = signedIntToBignum(GINTEGER(y));
	      temp = bigx;
	      bigx = bigsum(bigx, bigy);
	      FREE_BIGNUM(temp);
	      FREE_BIGNUM(bigy);
	      result = bignum_to_ground(bigx);
	      FREE_BIGNUM(bigx);
	      return result;
	    }
	  else return integer_to_ground(result);
	}
	case float_ground:
	  return float_to_ground(GINTEGER(x) + GFLOAT(y));
	case bignum_ground: {
	  BignumType bigx, isum;
	  Grounding result;
	  bigx = signedIntToBignum(GINTEGER(x));
	  isum = bigsum(bigx, GBIGNUM(y));
	  FREE_BIGNUM(bigx);
	  result = bignum_to_ground(isum);
	  FREE_BIGNUM(isum);
	  return result;
	}
	case rational_ground: {
	  Grounding iproduct, isum, result;
	  iproduct = generic_times(x, GCDR(y));
	  isum = generic_plus(iproduct, GCAR(y));
	  USE_GROUND(isum); FREE_GROUND(iproduct); 
	  result = make_rational(isum, GCDR(y));
	  FREE_GROUND(isum);
	  return result;
	}
	default:
	  ground_error(Type_Error,"not a number: ",y);
	  return NULL;
	}
    case (float_ground):
      switch (ground_type(y))
	{
	case float_ground:
	  return float_to_ground(GFLOAT(x) + GFLOAT(y));
	case integer_ground:
	  return float_to_ground(GFLOAT(x) + GINTEGER(y));
	case bignum_ground: {
	  Grounding ratx, result;
	  ratx = FloatToRational(x);
	  result = generic_plus(ratx, y);
	  FREE_GROUND(ratx);
	  return result;
	}
	case rational_ground:  {
	  Grounding iproduct, isum, result;
	  iproduct = generic_times(x, GCDR(y));
	  isum = generic_plus(iproduct, GCAR(y));
	  FREE_GROUND(iproduct);
	  result = generic_divide(isum, GCAR(y));
	  FREE_GROUND(isum);
	  return result;
	}
	default:
	  ground_error(Type_Error,"not a number: ",y);
	  return NULL;
	}
    case (bignum_ground):
      switch (ground_type(y))
	{
	case bignum_ground: {
	  BignumType isum;
	  Grounding result;
	  isum = bigsum(GBIGNUM(x), GBIGNUM(y));
	  result = bignum_to_ground(isum);
	  FREE_BIGNUM(isum);
	  return result;
	}
	case integer_ground: {
	  BignumType bigy, isum;
	  Grounding result;
	  bigy = signedIntToBignum(GINTEGER(y));
	  isum = bigsum(GBIGNUM(x), bigy);
	  FREE_BIGNUM(bigy);
	  result = bignum_to_ground(isum);
	  FREE_BIGNUM(isum);
	  return result;
	}
	case float_ground: {
	  Grounding raty = FloatToRational(y);
	  Grounding result = generic_plus(x, raty);
	  FREE_GROUND(raty);
	  return result;
	}
	case rational_ground: {
	  Grounding iproduct, isum, result;
	  iproduct = generic_times(x,GCDR(y));
	  isum = generic_plus(iproduct, GCAR(y));
	  USE_GROUND(isum); FREE_GROUND(iproduct);
	  result = make_rational(isum, GCDR(y));
	  FREE_GROUND(isum);
	  return result;
	}
	default:
	  ground_error(Type_Error, "not a number: ", y);
	  return NULL;
	}
    case (rational_ground):
      switch (ground_type(y))
	{
	case rational_ground: {
	  Grounding iproduct1, iproduct2, inumer, idenom, result;
	  iproduct1 = generic_times(GCAR(x), GCDR(y));
	  iproduct2 = generic_times(GCAR(y), GCDR(x));
	  inumer = generic_plus(iproduct1, iproduct2);
	  FREE_GROUND(iproduct1);
	  FREE_GROUND(iproduct2);
	  idenom = generic_times(GCDR(x), GCDR(y));
	  result = make_rational(inumer, idenom);
	  FREE_GROUND(inumer);
	  FREE_GROUND(idenom);
	  return result;
	}
	case integer_ground:
	case bignum_ground: {
	  Grounding iproduct, isum, result;
	  iproduct = generic_times(GCDR(x),y);
	  isum = generic_plus(GCAR(x), iproduct);
	  USE_GROUND(isum); FREE_GROUND(iproduct);
	  result = make_rational(isum, GCDR(x));
	  FREE_GROUND(isum);
	  return result;
	}	  
	case float_ground: {
	  Grounding iproduct, isum, result;
	  iproduct = generic_times(GCDR(x), y);
	  isum = generic_plus(GCAR(x), iproduct);
	  FREE_GROUND(iproduct);
	  result = generic_divide(isum, GCDR(x));
	  FREE_GROUND(isum);
	  return result;
	}
	default:
	  ground_error(Type_Error,"not a number: ",y);
	  return NULL;
	}
    default:			/* unknown ground_type(x) */
      ground_error(Type_Error,"not a number: ",x);
      return NULL;
    }
}


Grounding generic_minus(Grounding minuend, Grounding subtrahend)
{
  Grounding negaddend, result;
  negaddend = generic_times(a_neg_one, subtrahend);
  result = generic_plus(minuend, negaddend);
  FREE_GROUND(negaddend);
  return result;
}
     

Grounding generic_times(Grounding x, Grounding y)
{
  switch (ground_type(x))
    {
    case (integer_ground):
      switch (ground_type(y))
	{
	case integer_ground:  {
	  double result = (double) GINTEGER(x) * GINTEGER(y);
	  if ((result > INT_MAX) || (result < INT_MIN)) {
	    Grounding result;
	    BignumType bigx = signedIntToBignum(GINTEGER(x));
	    BignumType bigy = signedIntToBignum(GINTEGER(y));
	    bigx = bigprod(bigx, bigy);
	    FREE_BIGNUM(bigy);
	    result = bignum_to_ground(bigx);
	    FREE_BIGNUM(bigx);
	    return result;
	  }
	  else return integer_to_ground(result);
	}
	case float_ground:  
	  return float_to_ground(GINTEGER(x) * GFLOAT(y));
	case bignum_ground: {
	  BignumType bigx;
	  Grounding result;
	  bigx = signedIntToBignum(GINTEGER(x));
	  bigx = bigprod(bigx, GBIGNUM(y));
	  result = bignum_to_ground(bigx);
	  FREE_BIGNUM(bigx);
	  return result;
	}
	case rational_ground: {
	  Grounding iproduct, result;
	  iproduct = generic_times(x, GCAR(y)); USE_GROUND(iproduct);
	  result = make_rational(iproduct, GCDR(y));
	  FREE_GROUND(iproduct);
	  return result;
	}
	default:
	  ground_error(Type_Error,"not a number: ",y);
	  return NULL;
	}
    case (float_ground):
      switch (ground_type(y))
	{
	case float_ground:
	  return float_to_ground(GFLOAT(x) * GFLOAT(y));
	case integer_ground:
	  return float_to_ground(GFLOAT(x) * GINTEGER(y));
	case bignum_ground: {
	  Grounding ratx, result;
	  ratx = FloatToRational(x);
	  result = generic_times(ratx, y);
	  FREE_GROUND(ratx);
	  return result;
	}
	case rational_ground: {
	  Grounding iproduct, result;
	  iproduct = generic_times(x, GCAR(y));
	  result = generic_divide(iproduct, GCDR(y));
	  FREE_GROUND(iproduct);
	  return result;
	}
	default:
	  ground_error(Type_Error,"not a number: ",y);
	  return NULL;
	}
    case (bignum_ground):
      switch (ground_type(y))
	{
	case bignum_ground: {
	  Grounding result;
	  BignumType prod = bigprod(GBIGNUM(x), GBIGNUM(y));
	  result = bignum_to_ground(prod);
	  FREE_BIGNUM(prod);
	  return result;
	}
	case integer_ground: {
	  BignumType bigy;
	  Grounding result;
	  bigy = signedIntToBignum(GINTEGER(y));
	  bigy = bigprod(GBIGNUM(x), bigy);
	  result = bignum_to_ground(bigy);
	  FREE_BIGNUM(bigy);
	  return result;
	}
	case float_ground: {
	  Grounding result, raty = FloatToRational(y);
	  result = generic_times(x, raty);
	  FREE_GROUND(raty);
	  return result;
	}
	case rational_ground: {
	  Grounding iproduct, result;
	  iproduct = generic_times(x, GCAR(y)); USE_GROUND(iproduct);
	  result = make_rational(iproduct, GCDR(y));
	  FREE_GROUND(iproduct);
	  return result;
	}
	default:
	  ground_error(Type_Error,"not a number: ",y);
	  return NULL;
	}
    case (rational_ground):
      switch (ground_type(y))
	{
	case rational_ground: {
	  Grounding numer, denom, result;
	  numer = generic_times(GCAR(x), GCAR(y));
	  denom = generic_times(GCDR(x), GCDR(y));
	  USE_GROUND(numer); USE_GROUND(denom);
	  result = make_rational(numer, denom);
	  FREE_GROUND(numer); FREE_GROUND(denom);
	  return result;
	}
	case integer_ground:
	case bignum_ground: {
	  Grounding iproduct, result;
	  iproduct = generic_times(GCAR(x), y); USE_GROUND(iproduct);
	  result = make_rational(iproduct, GCDR(x));
	  FREE_GROUND(iproduct);
	  return result;
	}
	case float_ground: {
	  Grounding raty, result;
	  raty = FloatToRational(y);
	  result = generic_times(x, raty);
	  FREE_GROUND(raty);
	  return result;
	}
	default:
	  ground_error(Type_Error,"not a number: ",y);
	  return NULL;
	}
    default: /* unknown ground_type(x) */
      ground_error(Type_Error,"not a number: ",x);
      return NULL;
    }
}

Grounding generic_equal(Grounding x, Grounding y);


Grounding FloatToRational(Grounding MrFloat)
/*Changes a float_ground to a rational_ground*/
{
  BignumType denom, numer, bigtemp;
  float MsFloat = GFLOAT(MrFloat);
  int temp;
  Grounding n, d, r;

  denom = signedIntToBignum(1);
  temp = (int) MsFloat;
  numer = signedIntToBignum(temp);
  MsFloat -= temp;
  while (MsFloat) {
    bigtemp = numer;
    numer = multBignumByInt(numer, 10);
    FREE_BIGNUM(bigtemp);
    bigtemp = denom;
    denom = multBignumByInt(denom, 10);
    FREE_BIGNUM(bigtemp);
    MsFloat *= 10;
    temp = (int) MsFloat;
    MsFloat -= temp;
    bigtemp = numer;
    numer = addIntToBignum(numer, temp);
    FREE_BIGNUM(bigtemp);
  }
  n = bignum_to_ground(numer);
  d = bignum_to_ground(denom);
  FREE_BIGNUM(numer);
  FREE_BIGNUM(denom);
  USE_GROUND(n); USE_GROUND(d);   /* ?????????????????????????????? */
  r = make_rational(n, d); 
  FREE_GROUND(n); FREE_GROUND(d); 
  return r;
}


Grounding generic_divide(Grounding x, Grounding y)
{
  if (generic_equal(y, a_zero)) {
    ground_error(Division_By_Zero, "trying to divide number by zero: ", x);
    return NULL;
  }
  switch (ground_type(x))
    {
    case (integer_ground):
      switch (ground_type(y))
	{
	case integer_ground:
	case bignum_ground:
	  return make_rational(x, y);
	case float_ground:
	  return float_to_ground(GINTEGER(x) / GFLOAT(y));
	case rational_ground: {
	  Grounding result, iproduct;
	  iproduct = generic_times(x, GCDR(y)); USE_GROUND(iproduct);
	  result = make_rational(iproduct, GCAR(y));
	  FREE_GROUND(iproduct);
	  return result;
	}
	default:
	  ground_error(Type_Error,"not a number: ",y);
	  return NULL;
	}
    case (float_ground):
      switch (ground_type(y))
	{
	case float_ground:
	  return float_to_ground(GFLOAT(x) / GFLOAT(y));
	case integer_ground:
	  return float_to_ground(GFLOAT(x) / GINTEGER(y));
	case bignum_ground: {
	  Grounding result, ratx;
	  ratx = FloatToRational(x);
	  result = generic_divide(ratx, y);
	  FREE_GROUND(ratx);
	  return result;
	}
	case rational_ground: {
	  Grounding result, iproduct;
	  iproduct = generic_times(x, GCDR(y));
	  result = generic_divide(iproduct, GCAR(y));
	  FREE_GROUND(iproduct);
	  return result;
	}
	default:
	  ground_error(Type_Error,"not a number: ",y);
	  return NULL;
	}
    case (bignum_ground):
      switch (ground_type(y))
	{
	case bignum_ground:
	case integer_ground:
	  return make_rational(x, y);
	case float_ground: {
	  Grounding result, raty;
	  raty = FloatToRational(y);
	  result = generic_divide(x, raty);
	  FREE_GROUND(raty);
	  return result;
	}
	case rational_ground: {
	  Grounding iproduct, result;
	  iproduct = generic_times(x, GCDR(y)); USE_GROUND(iproduct);
	  result = make_rational(iproduct, GCAR(y));
	  FREE_GROUND(iproduct);
	  return result;
	}
	default:
	  ground_error(Type_Error,"not a number: ",y);
	  return NULL;
	}
    case (rational_ground):
      switch (ground_type(y))
	{
	case rational_ground: {
	  Grounding numer, denom, result;
	  numer = generic_times(GCAR(x), GCDR(y));
	  denom = generic_times(GCDR(x), GCAR(y));
	  USE_GROUND(numer); USE_GROUND(denom);
	  result = make_rational(numer, denom);
	  FREE_GROUND(numer); FREE_GROUND(denom);
	  return result;
	}
	case integer_ground:
	case bignum_ground: {
	  Grounding iproduct, result;
	  iproduct = generic_times(GCDR(x), y); USE_GROUND(iproduct);
	  result = make_rational(GCAR(x), iproduct);
	  FREE_GROUND(iproduct);
	  return result;
	}
	case float_ground: {
	  Grounding iproduct, result;
	  iproduct = generic_times(GCDR(x), y); USE_GROUND(iproduct);
	  result = generic_divide(GCAR(x), iproduct);
	  FREE_GROUND(iproduct);
	  return result;
	}
	default:
	  ground_error(Type_Error,"not a number: ",y);
	  return NULL;
	}
    default: /* unknown ground_type(x) */
      ground_error(Type_Error,"not a number: ",x);
      return NULL;
    }
}

Grounding generic_div(Grounding x, Grounding y)
{
  if (generic_equal(y, a_zero)) {
    ground_error(Division_By_Zero, "trying to divide number by zero: ", x);
    return NULL;
  }
  switch (ground_type(x))
    {
    case (integer_ground):
      switch (ground_type(y))
	{
	case integer_ground:
	  return integer_to_ground(GINTEGER(x) / GINTEGER(y));
	case bignum_ground: {
	  BignumType bigx, quot;
	  Grounding result;
	  bigx = signedIntToBignum(GINTEGER(x));
	  quot = bigdiv(bigx, GBIGNUM(y));
	  FREE_BIGNUM(bigx);
	  result = bignum_to_ground(quot);
	  FREE_BIGNUM(quot);
	  return result;
	}
	case rational_ground:
	case float_ground:
	  ground_error(Type_Error, "integer only operation: ", y);
	  return NULL;
	default:
	  ground_error(Type_Error,"not a number: ",y);
	  return NULL;
	}
    case (bignum_ground):
      switch (ground_type(y))
	{
	case bignum_ground: {
	  BignumType quot = bigdiv(GBIGNUM(x), GBIGNUM(y));
	  Grounding result = bignum_to_ground(quot);
	  FREE_BIGNUM(quot);
	  return result;
	}
	case integer_ground: {
	  BignumType quot, bigy = signedIntToBignum(GINTEGER(y));
	  Grounding result;
	  quot = bigdiv(GBIGNUM(x), bigy);
	  FREE_BIGNUM(bigy);
	  result = bignum_to_ground(quot);
	  FREE_BIGNUM(quot);
	  return result;
	}
	case rational_ground:
	case float_ground:
	  ground_error(Type_Error, "integer only operation: ", y);
	  return NULL;
	default:
	  ground_error(Type_Error,"not a number: ",y);
	  return NULL;
	}
    case (rational_ground):
    case (float_ground):
      ground_error(Type_Error, "integer only operation: ", x);
      return NULL;
    default: /* unknown ground_type(x) */
      ground_error(Type_Error,"not a number: ",x);
      return NULL;
    }
}

Grounding generic_mod(Grounding x, Grounding y)
{
  if (generic_equal(y, a_zero)) {
    ground_error(Division_By_Zero, "trying to divide by zero: ", x);
    return NULL;
  }
  switch (ground_type(x))
    {
    case (integer_ground):
      switch (ground_type(y))
	{
	case integer_ground:
	  return integer_to_ground(GINTEGER(x) % GINTEGER(y));
	case bignum_ground: {
	  BignumType mod, bigx = signedIntToBignum(GINTEGER(x));
	  Grounding result;
	  mod = bigmod(bigx, GBIGNUM(y));
	  FREE_BIGNUM(bigx);
	  result = bignum_to_ground(mod);
	  FREE_BIGNUM(mod);
	  return result;
	}
	case rational_ground:
	case float_ground:
	  ground_error(Type_Error, "integer only operation: ", y);
	  return NULL;
	default:
	  ground_error(Type_Error,"not a number: ",y);
	  return NULL;
	}
    case (bignum_ground):
      switch (ground_type(y))
	{
	case bignum_ground: {
	  BignumType mod = bigmod(GBIGNUM(x), GBIGNUM(y));
	  Grounding result = bignum_to_ground(mod);
	  FREE_BIGNUM(mod);
	  return result;
	}
	case integer_ground: {
	  BignumType mod, bigy = signedIntToBignum(GINTEGER(y));
	  Grounding result;
	  mod = bigmod(GBIGNUM(x), bigy);
	  FREE_BIGNUM(bigy);
	  result = bignum_to_ground(mod);
	  FREE_BIGNUM(mod);
	  return result;
	}
	case rational_ground:
	case float_ground:
	  ground_error(Type_Error, "integer only operation: ", y);
	  return NULL;
	default:
	  ground_error(Type_Error,"not a number: ",y);
	  return NULL;
	}
    case (rational_ground):
    case (float_ground):
      ground_error(Type_Error, "integer only operation: ", x);
      return NULL;
    default: /* unknown ground_type(x) */
      ground_error(Type_Error,"not a number: ",x);
      return NULL;
    }
}

/* Generic comparison operations */


#define NATIVE_COMPARE(xt,x,comp,yt,y) if (xt(x) comp yt(y)) return x;     \
                                       else return NULL;
/* and, for comparisons using functions rather than operators: */
#define FUNC_COMPARE(x, comp, y, xground)    (comp(x, y)) ?  xground : NULL;

Grounding generic_greater_than(Grounding x, Grounding y)
{
  switch ground_type(x)
    {
    case integer_ground:
      switch ground_type(y)
	{
	case integer_ground: 
	  NATIVE_COMPARE(GINTEGER, x, >, GINTEGER, y);
	case float_ground:
	  NATIVE_COMPARE(GINTEGER, x, >, GFLOAT, y);
	case bignum_ground: {
	  BignumType bignum = signedIntToBignum(GINTEGER(x));
	  Grounding result, ground = bignum_to_ground(bignum);
	  FREE_BIGNUM(bignum);
	  if (generic_greater_than(ground, y))
	    result = x;
	  else result = NULL;
	  FREE_GROUND(ground);
	  return result;
	}
	case rational_ground: {
	  Grounding iproduct, result;
	  iproduct = generic_times(x, GCDR(y));
	  result = FUNC_COMPARE(iproduct, generic_greater_than, GCAR(y), x);
	  FREE_GROUND(iproduct);
	  return result;
	}
	default:
	  ground_error(Type_Error,"not a number: ",y);
	  return NULL;
	}
    case float_ground:
      switch ground_type(y)
	{
	case float_ground:
	  NATIVE_COMPARE(GFLOAT, x, >, GFLOAT, y);
	case integer_ground: 
	  NATIVE_COMPARE(GFLOAT, x, >, GINTEGER, y);
	case bignum_ground: {
	  Grounding result, rat = FloatToRational(x);
	  if (generic_greater_than(rat, y))
	    result = x;
	  else result = NULL;
	  FREE_GROUND(rat);
	  return result;
	} 
	case rational_ground: {
	  Grounding iproduct, result;
	  iproduct = generic_times(x, GCDR(y));
	  result = FUNC_COMPARE(iproduct, generic_greater_than, GCAR(y), x);
	  FREE_GROUND(iproduct);
	  return result;
	}
	default:
	  ground_error(Type_Error,"not a number: ",y);
	  return NULL;
	}	  
    case bignum_ground:
      switch ground_type(y)
	{
	case bignum_ground:
	  return FUNC_COMPARE(GBIGNUM(x), biggreaterthan, GBIGNUM(y), x);
	case integer_ground:
	  if (((GINTEGER(y)) > 0) && (((GBIGNUM(x))->sign) < 0))
	    return x;
	  else return NULL;
	case float_ground: {
	  Grounding result, rat = FloatToRational(y);
	  if (generic_greater_than(x, rat))
	    result = x;
	  else result = NULL;
	  FREE_GROUND(rat);
	  return result;
	} 
	case rational_ground: {
	  Grounding iproduct, result;
	  iproduct = generic_times(x, GCDR(y));
	  result = FUNC_COMPARE(iproduct, generic_greater_than,
				GCAR(y), x);
	  FREE_GROUND(iproduct);
	  return result;
	}
	default:
	  ground_error(Type_Error,"not a number: ",y);
	  return NULL;
	}
    case rational_ground:
      switch ground_type(y)
	{
	case rational_ground:  {
	  Grounding iprod1, iprod2, result;
	  iprod1 = generic_times(GCAR(x), GCDR(y));
	  iprod2 = generic_times(GCAR(y), GCDR(x));
	  result = FUNC_COMPARE(iprod1, generic_greater_than, iprod2, x);
	  FREE_GROUND(iprod1); FREE_GROUND(iprod2);
	  return result;
	}
	case integer_ground:
	case float_ground:
	case bignum_ground:
	  {
	  Grounding iproduct, result;
	  iproduct = generic_times(y, GCDR(x));
	  result = FUNC_COMPARE(GCAR(x), generic_greater_than, iproduct, x);
	  FREE_GROUND(iproduct);
	  return result;
	}
	default:
	  ground_error(Type_Error,"not a number: ",y);
	  return NULL;
	}
    default: /* unknown ground_type(x) */
      ground_error(Type_Error,"not a number: ",x);
      return NULL;
    }
}


Grounding generic_equal(Grounding x, Grounding y)
{
  switch ground_type(x)
    {
    case integer_ground:
      switch ground_type(y)
	{
	case integer_ground: 
	  NATIVE_COMPARE(GINTEGER, x, ==, GINTEGER, y);
	case float_ground:
	  NATIVE_COMPARE(GINTEGER, x, ==, GFLOAT, y);
	case bignum_ground: {
	  BignumType bignum = signedIntToBignum(GINTEGER(x));
	  Grounding result, ground = bignum_to_ground(bignum);
	  FREE_BIGNUM(bignum);
	  if (generic_equal(ground, y))
	    result = x;
	  else result = NULL;
	  FREE_GROUND(ground);
	  return result;
	}
	case rational_ground: {
	  Grounding iproduct, result;
	  iproduct = generic_times(x, GCDR(y));
	  result = FUNC_COMPARE(iproduct, generic_equal, GCAR(y), x);
	  FREE_GROUND(iproduct);
	  return result;
	}
	default:
	  ground_error(Type_Error,"not a number: ",y);
	  return NULL;
	}
    case float_ground:
      switch ground_type(y)
	{
	case float_ground:
	  NATIVE_COMPARE(GFLOAT, x, ==, GFLOAT, y);
	case integer_ground: 
	  NATIVE_COMPARE(GFLOAT, x, ==, GINTEGER, y);
	case bignum_ground: {
	  Grounding result, rat = FloatToRational(x);
	  if (generic_equal(rat, y))
	    result = x;
	  else result = NULL;
	  FREE_GROUND(rat);
	  return result;
	} 
	case rational_ground: {
	  Grounding iproduct, result;
	  iproduct = generic_times(x, GCDR(y));
	  result = FUNC_COMPARE(iproduct, generic_equal, GCAR(y), x);
	  FREE_GROUND(iproduct);
	  return result;
	}
	default:
	  ground_error(Type_Error,"not a number: ",y);
	  return NULL;
	}	  
    case bignum_ground:
      switch ground_type(y)
	{
	case bignum_ground:
	  return FUNC_COMPARE(GBIGNUM(x), bigequal, GBIGNUM(y), x);
	case integer_ground:
	  return NULL;
	case float_ground: {
	  Grounding result, rat = FloatToRational(y);
	  if (generic_equal(x, rat))
	    result = x;
	  else result = NULL;
	  FREE_GROUND(rat);
	  return result;
	} 
	case rational_ground: {
	  Grounding iproduct, result;
	  iproduct = generic_times(x, GCDR(y));
	  result = FUNC_COMPARE(iproduct, generic_equal,
				GCAR(y), x);
	  FREE_GROUND(iproduct);
	  return result;
	}
	default:
	  ground_error(Type_Error,"not a number: ",y);
	  return NULL;
	}
    case rational_ground:
      switch ground_type(y)
	{
	case rational_ground:  {
	  Grounding iprod1, iprod2, result;
	  iprod1 = generic_times(GCAR(x), GCDR(y));
	  iprod2 = generic_times(GCAR(y), GCDR(x));
	  result = FUNC_COMPARE(iprod1, generic_equal, iprod2, x);
	  FREE_GROUND(iprod1); FREE_GROUND(iprod2);
	  return result;
	}
	case integer_ground:
	case float_ground:
	case bignum_ground:
	  {
	  Grounding iproduct, result;
	  iproduct = generic_times(y, GCDR(x));
	  result = FUNC_COMPARE(GCAR(x), generic_equal, iproduct, x);
	  FREE_GROUND(iproduct);
	  return result;
	}
	default:
	  ground_error(Type_Error,"not a number: ",y);
	  return NULL;
	}
    default: /* unknown ground_type(x) */
      ground_error(Type_Error,"not a number: ",x);
      return NULL;
    }
}


Grounding generic_less_than(Grounding x, Grounding y)
{
/*short code version*/
  Grounding ygtx; ygtx = generic_greater_than(y, x);
  if (ygtx) return x; else return NULL;
}


Grounding generic_gcd(Grounding x, Grounding y)
{
  switch (ground_type(x))
    {
    case (integer_ground):
      switch (ground_type(y))
	{
	case integer_ground:
	  return integer_to_ground(intGCD(GINTEGER(x), GINTEGER(y)));
	case bignum_ground: {
	  BignumType gcd, bigx = signedIntToBignum(GINTEGER(x));
	  Grounding result;
	  gcd = bigGCD(bigx, GBIGNUM(y));
	  FREE_BIGNUM(bigx);
	  result = bignum_to_ground(gcd);
	  FREE_BIGNUM(gcd);
	  return result;
	}
	case float_ground:
	case rational_ground:
	  ground_error(Type_Error, "integer only operation: ", y);
	  return NULL;
	default:
	  ground_error(Type_Error,"not a number: ", y);
	  return NULL;
	}
    case (bignum_ground):
      switch (ground_type(y))
	{
	case bignum_ground: {
	  BignumType gcd = bigGCD(GBIGNUM(x), GBIGNUM(y));
	  Grounding result = bignum_to_ground(gcd);
	  FREE_BIGNUM(gcd);
	  return result;
	}
	case integer_ground: {
	  BignumType gcd, bigy = signedIntToBignum(GINTEGER(y));
	  Grounding result;
	  gcd = bigGCD(GBIGNUM(x), bigy);
	  FREE_BIGNUM(bigy);
	  result = bignum_to_ground(gcd);
	  FREE_BIGNUM(gcd);
	  return result;
	}
	case float_ground:
	case rational_ground:
	  ground_error(Type_Error,"integer only operation: ", y);
	  return NULL;
	default:
	  ground_error(Type_Error,"not a number: ", y);
	  return NULL;
	}
    case (rational_ground):
    case (float_ground):
      ground_error(Type_Error,"integer only operation: ", y);
      return NULL;
    default: /* unknown ground_type(x) */
      ground_error(Type_Error,"integer only operation: ",x);
      return NULL;
    }
}

Grounding arithmetic_lexpr(Grounding (*binary_op)(),Grounding args)
{
  /* This is slightly hairy because we need to garbage collect
     intermediate results. */
  Grounding result=GCAR(args), remainder=GCDR(args);
  if (remainder == empty_list) return result;
  else {USE_GROUND(result);
	{DO_LIST(num,GCDR(args))
	   {Grounding old_result; old_result=result;
	    result=binary_op(result,GCAR(remainder)); 
	    if (result != old_result) 
	      {FREE_GROUND(old_result); USE_GROUND(result);}}}}
  ref_count(result)--;
  return result;
}

Grounding fraxl_plus_lexpr(Grounding numbers)
{
  return arithmetic_lexpr(generic_plus,numbers);
}


Grounding fraxl_minus_lexpr(Grounding numbers) 
{
  if (GCDR(numbers) == empty_list)
    return generic_times(a_neg_one, GCAR(numbers));
  return arithmetic_lexpr(generic_minus,numbers);
}

Grounding fraxl_times_lexpr(Grounding numbers)
{
  return arithmetic_lexpr(generic_times,numbers);
}

Grounding fraxl_div_lexpr(Grounding numbers) 
{
  return arithmetic_lexpr(generic_div,numbers);
}

Grounding fraxl_divide_lexpr(Grounding numbers) 
{
  if (GCDR(numbers) == empty_list)
    return generic_divide(a_one, GCAR(numbers));
  else return arithmetic_lexpr(generic_divide,numbers);
}

Grounding fraxl_random(Grounding max)
{
  int choice, range; 
  choice=rand(); range=GINTEGER(max);
  return integer_to_ground(choice%range);
}



/* Bignums */

/**************************************************
 Functions for operations on bignums
**************************************************/


Grounding bignum_to_ground(BignumType big)
{
  Grounding result;
  if (big->sign == 0) return integer_to_ground(0);
  else if (BITS_IN(BIGDIGIT_TYPE) * big->numdigits <= BITS_IN(int))
    {
      int temp;
      temp=bignumToInt(big);
      return integer_to_ground(temp);
    }
  else
    {
      BignumType carboncopy = copyBignum(big);
/*List stuff...
      bignum_list *temp = fra_allocate(1,sizeof(bignum_list));
      temp->ptrcount = 1;
      temp->initline = __LINE__;
      temp->bignum = carboncopy;
      temp->next = biglist;
      biglist = temp;
*/    
      INITIALIZE_NEW_GROUND(result, bignum_ground);
      the(bignum,result) = carboncopy;
      return result;
    }
}

#if 0
int noDuplicates(BignumType bn, bignum_list *listptr)
/*returns true if bignum is not in biglist after (not including) listptr */
{
  int nodupes = 1;
  while (listptr->next) {
    listptr = listptr->next;
    if (listptr->bignum == bn) {
      nodupes = 0;
      break;
    }
  }
  return nodupes;
}
#endif

static BignumType initializeBignum(int linenum) {
/*  struct BIGNUM_LIST *temp =
    (bignum_list *) fra_allocate(1,sizeof(bignum_list));
*/
  BignumType newnum =
    (BignumType) fra_allocate(1, sizeof(struct BIGNUM_STRUCT));
  newnum->sign = 0;
  newnum->numdigits = 1;
/*
  temp->next = biglist;
  temp->bignum = newnum;
  temp->initline = linenum;
  temp->ptrcount = 1;
  biglist = temp;
*/
  newnum->digits[0] = 0;
  debug(z++);
  return newnum;
}

static BignumType grow_bignum(BignumType bignum, int by)
/* Does NOT fill in with zeroes*/
{
/*BignumType temp = bignum;
*/
  bignum->numdigits += by;
  bignum = (BignumType) fra_reallocate(bignum, sizeof(struct BIGNUM_STRUCT)+ 
				((bignum->numdigits-1)*sizeof(BIGDIGIT_TYPE)));
/*if (bignum != temp) {
    bignum_list *runner = biglist;
    while (runner->bignum != temp) runner = runner->next;
    runner->bignum = bignum;
  }
*/
  return bignum;
}

#if !defined(GROW_BIGNUM)
#define GROW_BIGNUM(bignum, by) bignum = grow_bignum(bignum, by)
#endif


/* bignum-bignum math */

BignumType cleanUpBignum(BignumType bignum)
/*eliminates leading zeroes */
{
  unsigned long int start=bignum->numdigits;
  
  while ((bignum->digits[bignum->numdigits - 1] == 0)&&(bignum->numdigits > 1))
    bignum->numdigits--;
  if (bignum->numdigits!=start)
    GROW_BIGNUM(bignum,0);  /*growing by 0 alloc's numdigits digits*/
  if ((bignum->numdigits <= 1) && (bignum->digits[0] == 0)) {
    bignum->sign = 0;
    bignum->numdigits = 1;
  }
  return bignum;
}

BignumType copyBignum(BignumType from)
{
  BignumType to = initializeBignum(__LINE__);
  unsigned long int digitCount=0;
  BIGDIGIT_TYPE *fromptr;
  BIGDIGIT_TYPE *toptr;
  
  GROW_BIGNUM(to, from->numdigits - 1);
  to->sign=from->sign;
  to->numdigits=from->numdigits;
  fromptr=from->digits;
  toptr=to->digits;
  for (digitCount=0; digitCount < from->numdigits;
       digitCount++, fromptr++, toptr++)
    *toptr = *fromptr;
  return(to);
}


signed char bignumCompareDgts(BignumType bignum1, BignumType bignum2) 
/* returns 1 if |bignum1| is larger,
   0 if equal, -1 if |bignum2| is larger. */
/* removes leading zeroes from bignums */
{
  unsigned long int digitCount;
  BIGDIGIT_TYPE *digitptr1, *digitptr2;

  cleanUpBignum(bignum1);
  cleanUpBignum(bignum2);
  if (bignum1->numdigits > bignum2->numdigits) return 1;
  else if (bignum1->numdigits < bignum2->numdigits) return -1;
  else {
    digitCount=bignum1->numdigits;
    digitptr1 = &(bignum1->digits[digitCount-1]);
    digitptr2 = &(bignum2->digits[digitCount-1]);
    while ((digitptr1 > bignum1->digits) && (*digitptr1==*digitptr2)) {
      digitptr1--;
      digitptr2--; }
    if (*digitptr1 > *digitptr2) return 1;
    else if (*digitptr1 < *digitptr2) return -1;
    else return 0;
  }
}

char biglessthan(BignumType num1, BignumType num2)
{
  return ((num1->sign < num2->sign) ||
	  ((num1->sign == num2->sign) &&
	   (bignumCompareDgts(num1, num2) == -1)));
}

char biggreaterthan(BignumType num1, BignumType num2)
{
  return ((num1->sign > num2->sign) ||
	  ((num1->sign == num2->sign) &&
	   (bignumCompareDgts(num1, num2) == 1)));
}


char bigequal(BignumType num1, BignumType num2)
{
  return ((bignumCompareDgts(num1, num2) == 0) && (num1->sign == num2->sign) );
}

BignumType addSameSign(BignumType bigaddend, BignumType smalladdend)
/* larger addend must be first if lengths differ */
{
  BignumType result;
  unsigned long int tempsum, carry=0;
  unsigned long int digitCount;

  result = initializeBignum(__LINE__);
  GROW_BIGNUM(result, bigaddend->numdigits - 1);
  for (digitCount = 0; digitCount<smalladdend->numdigits ; digitCount++) {
    tempsum = carry + smalladdend->digits[digitCount] +
      bigaddend->digits[digitCount];
    result->digits[digitCount] = (BIGDIGIT_TYPE) tempsum;
    carry = tempsum >> BITS_IN(BIGDIGIT_TYPE);
  }
  for (;digitCount<bigaddend->numdigits; digitCount++) {
    tempsum = carry + bigaddend->digits[digitCount];
    result->digits[digitCount] = (BIGDIGIT_TYPE) tempsum;
    carry = tempsum >> BITS_IN(BIGDIGIT_TYPE);
  }
  if (carry!=0) {
    /* printf("addSameSign: reallocating\n"); */  /*Is there a bug here?*/
    GROW_BIGNUM(result,1);
    result->digits[digitCount] = (BIGDIGIT_TYPE) carry;
    carry /= RADIX(BIGDIGIT_TYPE);
    digitCount++;
  }
  result->numdigits=digitCount;
  result->sign=bigaddend->sign;
  return result;
}

BignumType subtrDigitFromBignum(BignumType bignum, int digit, int offset)
/* Used by addOppSigns, affects bignum */
{
  if (bignum->numdigits <= offset) {
    ground_error(Func_Error, "subtrDigitFromBignum: bignum too small: ",
		 NULL);
    return bignum;
  } 
  if (digit <= bignum->digits[offset]) {
    bignum->digits[offset] -= digit;
    bignum = cleanUpBignum(bignum);
    return bignum;
  }
  else {
    if (bignum->numdigits <= offset+1)
      ground_error(Func_Error, "subtrDigitFromBignum: bignum too small: ",
		 NULL);
    bignum->digits[offset] += (RADIX(BIGDIGIT_TYPE) - digit); 
    return (subtrDigitFromBignum(bignum, 1, offset+1));
  }
}

BignumType addOppSigns(BignumType bigaddend, BignumType smalladdend)
{
  unsigned long int digitCount;
  BignumType result;

  result = copyBignum(bigaddend);
  for (digitCount=0; digitCount<smalladdend->numdigits; digitCount++)
    result = subtrDigitFromBignum(result,
				  smalladdend->digits[digitCount],
				  digitCount);   
  return result;
}


BignumType bigsum(BignumType addend1, BignumType addend2)
{
  BignumType smalladdend, bigaddend, result;
  
  if (addend1->sign==0) return copyBignum(addend2);
  if (addend2->sign==0) return copyBignum(addend1);
  switch (bignumCompareDgts(addend1, addend2))
    {
    case (0):  /* |addend1| == |addend2|*/
      if (addend1->sign != addend2->sign) {
	return (initializeBignum(__LINE__)); }
      else {
	bigaddend = addend1;
	smalladdend = addend2; }
    case (1): /* |addend1| > |addend2| */
      bigaddend = addend1;
      smalladdend = addend2;
      break;
    case (-1): /* |addend1| < |addend2| */
      bigaddend = addend2;
      smalladdend = addend1;
      break;
    }
  if (smalladdend->sign == bigaddend->sign)
    result = addSameSign(bigaddend, smalladdend);
  else
    result = addOppSigns(bigaddend, smalladdend);
  FREE_BIGNUM(bigaddend);
  FREE_BIGNUM(smalladdend);
  return result;
}



BignumType bigdiff(BignumType minuend, BignumType subtrahend) 
/* minuend - subtrahend */
{
  BignumType result, negaddend = copyBignum(subtrahend);
  negaddend->sign *= -1;
  result = bigsum(minuend, negaddend);
  FREE_BIGNUM(negaddend);
  return result;
}


static BignumType addIntToBignum(BignumType bignum, int addend)
  /*use nonnegative addend and bignum*/
  /*affects bignum*/
{
  BIGDIGIT_TYPE *digitptr, *endptr;
  unsigned long int tempsum, carry=addend;

  if ((! bignum->sign) && (addend)) bignum->sign = 1;
  digitptr = bignum->digits;
  endptr = &(bignum->digits[bignum->numdigits - 1]); /*the end of digits*/
  while ((carry != 0) && (digitptr <= endptr)) {
    tempsum = carry + *digitptr;
    *digitptr = (BIGDIGIT_TYPE) tempsum;
    carry = tempsum >> BITS_IN(BIGDIGIT_TYPE);
    digitptr ++;
  }
  while (carry != 0) { /*This never seems to happen */
    /*printf("\naddIntToBignum: reallocating\n");*/
    GROW_BIGNUM(bignum,1);
    /*Can't use digitptr now...it points to pre-realloc bignum*/
    bignum->digits[bignum->numdigits - 1] = (BIGDIGIT_TYPE) carry;
    carry = carry >> BITS_IN(BIGDIGIT_TYPE);
  }
  return bignum;
}			  
		  
static BignumType multBignumByInt(BignumType bignum, int factor)
/*No neg's. */
{
  BignumType result;
  int carry=0;
  BIGDIGIT_TYPE *digitptr, *endptr;
  unsigned long int tempProd;

  if (factor == 0) {
    result = initializeBignum(__LINE__);
    return result;
    }
  result = copyBignum(bignum);
  digitptr = result->digits;
  endptr = &(result->digits[result->numdigits-1]);
  while (digitptr <= endptr) {
    tempProd = factor * (*digitptr) + carry;
    *digitptr = (BIGDIGIT_TYPE) tempProd;
    carry = tempProd >> BITS_IN(BIGDIGIT_TYPE);
    digitptr++; }
  while (carry != 0) {  /* I could probably just use an if */
    /*printf("multBignumByInt: reallocating\n");*/
    GROW_BIGNUM(result,1);
    /* Can't use digitptr because it still points to pre-realloc result */
    result->digits[result->numdigits - 1] = (BIGDIGIT_TYPE) carry;
    carry = carry / RADIX(BIGDIGIT_TYPE);
  }
  return result;
}


BignumType multBignumByRadix(BignumType bignum)
/*alters bignum*/
{
  BIGDIGIT_TYPE *digitptr;
  
  GROW_BIGNUM(bignum,1);
  digitptr = &(bignum->digits[bignum->numdigits-1]);
  while (digitptr > bignum->digits) {
    *digitptr = *(digitptr -  1);
    digitptr--;
  }
  *digitptr = 0;
  return bignum;
}

BignumType bigprod(BignumType factor1, BignumType factor2)
/*digitwise method*/
{
  BignumType runningtotal = initializeBignum(__LINE__);
  BignumType tempfree, temp;
  BIGDIGIT_TYPE *digitptr;

/*first check for zero factor:*/
  if (!(factor1->sign * factor2->sign)) return runningtotal;
  digitptr = &( factor1->digits[ factor1->numdigits - 1 ] );
  runningtotal = multBignumByInt(factor2, *digitptr); 
  digitptr--;
  while (digitptr >= factor1->digits) {
    runningtotal = multBignumByRadix(runningtotal);
    temp = multBignumByInt(factor2, *digitptr);
    tempfree = runningtotal;
    runningtotal = bigsum(runningtotal, temp);
    FREE_BIGNUM(tempfree);
    FREE_BIGNUM(temp);
    digitptr--;
  }
  runningtotal->sign = factor1->sign * factor2->sign;
  return runningtotal;
}


/* These are used by the div and mod dispatchers: */
#ifndef QUOT
#define QUOT 'q'
#define MOD 'm'
#endif

BignumType divmodBignumByInt(BignumType bignum, int denom, char ret)
{
  /*dispatched to by bigdiv and bigmod, and used by longbigdivmod*/
  /*returns either quotient or remainder depending on ret (QUOT/MOD)*/
  /*ignores sign, doesn't check for division by zero*/
  
  unsigned long int temp;
  int digit;
  BIGDIGIT_TYPE *digitptr;
  BignumType remainder, quotient = initializeBignum(__LINE__);
  
  digitptr = &( bignum->digits[ bignum->numdigits - 1] ); /*most signif digit*/
  temp = *digitptr;
  digit = temp/denom;
  quotient = addIntToBignum(quotient, digit);
  while (digitptr > bignum->digits) {
    digitptr --;
    quotient = multBignumByRadix(quotient);
    temp -= digit * denom;
    temp *= RADIX(BIGDIGIT_TYPE);
    temp += *digitptr;
    digit = temp/denom;
    quotient = addIntToBignum(quotient, digit);
  }
  if (ret == MOD) {
    temp -= digit * denom;
    remainder = initializeBignum(__LINE__);
    remainder = addIntToBignum(remainder, temp);
    FREE_BIGNUM(quotient);
    return remainder;
  }
  else if (ret == QUOT)
    return quotient;
  else
    {ground_error(Func_Error,
		  "divmodBignumByInt: Improper call; quotient returned",
		  NULL);
     return NULL;}
   } 

BignumType copypart(BignumType bignum, unsigned long int start,
		    unsigned long int digits)
{
  /*used by longbigdivmod, returns a positive bignum*/
  BignumType to = initializeBignum(__LINE__);
  BIGDIGIT_TYPE *digitptr, *toptr;

  GROW_BIGNUM(to, digits - 1);
  digitptr = &(bignum->digits[start]);
  toptr = to->digits;
  for (; digits > 0; digitptr++, toptr++, digits--) *toptr = *digitptr;
  to->sign = 1;
  return to;
}

BignumType basetopow(unsigned long int pow)
{
  /*used by longbigdivmod*/
  /*returns a bignum that equals RADIX(BIGDIGITS_TYPE)^pow */
  BignumType result = initializeBignum(__LINE__);
  BIGDIGIT_TYPE *digitptr, *endptr;

  GROW_BIGNUM(result, pow);
  digitptr = result->digits;
  endptr = &( result->digits[ result->numdigits - 1] );
  while (digitptr < endptr) {
    *digitptr = 0;
    digitptr++;
  }
  *digitptr = 1;
  result->sign = 1;
  return result;
}
    
BignumType longbigdivmod(BignumType Numer, BignumType Denom, char ret)
{
  /*Algorithm adapted from Knuth, _Art_of_Computer_Programming_ Vol 2*/
  /*dispatched to by bigdiv and bigmod*/
  /*returns quotient or remainder depending on value of ret (QUOT/MOD)
    by default returns quotient */


  BignumType quotient = initializeBignum(__LINE__);
  BignumType remainder;
  BignumType numer = copyBignum(Numer);
  BignumType denom = copyBignum(Denom);
  BignumType partnum, tempprod, temp;
  unsigned long int i, j;
  unsigned long int n, m;
  unsigned long int b;
  BIGDIGIT_TYPE d, guess, *tempptr, *numerptr, *denomptr;

  if (! numer->sign)
    { d = 1;
      /*quotient stays at zero*/
      n = numer->numdigits;
    }
  else {
    b = RADIX(BIGDIGIT_TYPE);
    /* To make sure most signif digit in denom >= base/2 */
    n = denom->numdigits;
    m = (numer->numdigits - n);
/* Set normalizing factor, d, such that, after mult'ing numer and denom by d,
   most signif digit of denom >= b/2 */
    d = b / (denom->digits[denom->numdigits - 1] + 1);
    if (d != 1) {
      temp = numer;
      numer = multBignumByInt(numer, d);
      FREE_BIGNUM(temp);
      temp = denom;
      denom = multBignumByInt(denom, d);
      FREE_BIGNUM(temp);
    }
    if (n+m == numer->numdigits) { /* NUMER DIDN'T EXPAND...expand it now */
/* w/o changing value, tack a 0 onto numer */
      GROW_BIGNUM(numer, 1);
      numer->digits[numer->numdigits - 1] = 0;
    }
    GROW_BIGNUM(quotient, m);	/* quotient has m+1 digits */
    quotient->sign = numer->sign * denom->sign;
    numer->sign = denom->sign = 1;
    denomptr = &(denom->digits[n - 1]); /* most signif digit */
    for (j = 0; j <= m; j++) {
      numerptr = &(numer->digits[numer->numdigits - 1 - j]);
/* make an educated guess for the next digit of the quotient */
      if (*numerptr == *denomptr)  guess = b - 1;
      else  guess = ((*numerptr * b) + *(numerptr - 1)) / *denomptr;
      while ( (*(denomptr-1) * guess) >
	     ((*numerptr * b + *(numerptr-1) - guess * *denomptr) * b) +
	     *(numerptr-2))
	guess --;
      tempprod = multBignumByInt(denom, guess);
      /* get the significant digits of the numerator to work with */
      partnum = copypart(numer, (numer->numdigits - n - j - 1), n+1);
      if (bignumCompareDgts(tempprod, partnum) == 1) { /* tempprod>partnum */
	guess--;
	FREE_BIGNUM(tempprod);
	tempprod = multBignumByInt(denom, guess);
      }
      temp = partnum;
      partnum = bigdiff(partnum, tempprod);
      FREE_BIGNUM(temp);
      
      quotient->digits[ quotient->numdigits - 1 - j ] = guess;
      /* copy partnum back into appropriate section of numer */
      numerptr = &(numer->digits[numer->numdigits - 1 - n - j]);
      tempptr = partnum->digits;
      for (i=0; i<n+1; numerptr++, tempptr++, i++) {
	if (i < partnum->numdigits)
	  *numerptr = *tempptr;
	else *numerptr = 0;
      }
    }
    FREE_BIGNUM(tempprod);
    FREE_BIGNUM(partnum);
  }
  FREE_BIGNUM(denom);
  if (ret == MOD) {
    FREE_BIGNUM(quotient);
    remainder = copypart(numer, 0, n);
    FREE_BIGNUM(numer);
    remainder = cleanUpBignum(remainder);
    temp = remainder;
    remainder = divmodBignumByInt(remainder, d, QUOT);
    FREE_BIGNUM(temp);
    return(remainder);
  }
  else if (ret == QUOT) {
    FREE_BIGNUM(numer);
    return quotient; }
  else {ground_error(Func_Error, "longbigdivmod: Improper call", NULL);
	return NULL;}
}

  
BignumType bigdiv(BignumType numerator, BignumType denominator)
/*In non-trivial situations, dispatches to specialized functions*/
/*Returns quotient*/
{
  BignumType quotient = initializeBignum(__LINE__); /*init to zero*/

/* generic_divide checks for division by zero */
  if (numerator->sign == 0) return quotient; /* Numerator is zero */
  switch (bignumCompareDgts(numerator, denominator))
    {
    case (0): /* numerator == denominator */
      *quotient->digits = 1;
      quotient->sign = 1;
      break;
    case (-1): /*numerator < denominator */
      break; /* return init'd bignum */
    case (1): /* numerator > denominator */
      if (denominator->numdigits == 1) {
	if (numerator->numdigits == 1) {
	  *quotient->digits = *numerator->digits / *denominator->digits;
	  quotient->sign = numerator->sign * denominator->sign;
	  return quotient; /*just one digit*/
	}
	else { /* numerator has more than 1 digit, denom has 1 */
	  quotient = divmodBignumByInt(numerator, *denominator->digits, QUOT);
	  quotient->sign = numerator->sign * denominator->sign;
	  return quotient;
	}
      }
      else {  /* numerator and denominator have multiple digits */
	quotient = longbigdivmod(numerator, denominator, QUOT);
	break;
      }
    }
  return quotient;
}




BignumType bigmod(BignumType numerator, BignumType denominator)
/*In non-trivial situations, dispatches to specialized functions*/
/*Returns remainder*/
/* does not check for division by zero */
{
  BignumType remainder;

  /* Numerator is zero */
  if (numerator->sign == 0) return copyBignum(denominator);
  else remainder = initializeBignum(__LINE__); /*init to zero*/
  switch (bignumCompareDgts(numerator, denominator))
    {
    case (0): /* numerator == denominator */
      break; /*return 0 */
    case (-1): /* numerator < denominator */
      return copyBignum(numerator);
    case (1): /* numerator > denominator */
      if (denominator->numdigits == 1) {
	if (numerator->numdigits == 1) {
	  *remainder->digits = *numerator->digits % *denominator->digits;
	  if ((*remainder->digits) == 0) remainder->sign=0;
	  else remainder->sign = numerator->sign * denominator->sign;
	  return remainder; /*just one digit*/
	}
	else { /* numerator has more than 1 digit, denom has 1 */
	  remainder = divmodBignumByInt(numerator, *denominator->digits, MOD);
	  if (remainder->sign == -1) remainder->sign=1;
	  return remainder;
	}
      }
      else {  /* numerator and denominator have multiple digits */
	remainder = longbigdivmod(numerator, denominator, MOD);
	break;
      }
    }
  return remainder;
}


/****************************************************/
/* rational stuff */

int intGCD(int x, int y)
/*Modern Euclidian algorithm from Knuth _Art_of_Computer_Programming_ Vol 2 */
{
  int remainder;

  if (x<0) x *= -1;
  if (y<0) y *= -1;
  while (y) {
    remainder = x % y;
    x = y;
    y = remainder;
  }
  return x;
}
BignumType bigGCD(BignumType u, BignumType v)
/*Modern Euclidian algorithm from Knuth _Art_of_Computer_Programming_ Vol 2 */
{
  BignumType remainder;
  BignumType x = copyBignum(u);
  BignumType y = copyBignum(v);
  
  if (x->sign == -1) x->sign = 1;
  if (y->sign == -1) y->sign = 1;
  while (y->sign) {
    remainder = bigmod(x, y);
    FREE_BIGNUM(x);
    x = y;
    y = remainder;
  }
  FREE_BIGNUM(remainder);
  return x;
}

Grounding simplifyRational(Grounding rat)
{
  Grounding gcd, newcar, newcdr, result;

  if (! TYPEP(rat, rational_ground))
    ground_error(Func_Error, "simplifyRational was passed a non-rational",
		 rat);
  gcd = generic_gcd(GCAR(rat), GCDR(rat));
  newcar = generic_divide(GCAR(rat), gcd);
  newcdr = generic_divide(GCDR(rat), gcd);
  result = make_rational(newcar, newcdr);
  FREE_GROUND(newcar);
  FREE_GROUND(newcdr);
  return result;
}

/*********************************/
/* Conversion routines */


BignumType signedIntToBignum(signed int before)
{
  BignumType after = initializeBignum(__LINE__);

  if (before) {
    if (before < 0) {
      after->sign = -1;
      before *= -1;
    }
    after = addIntToBignum(after, before);
  }
  return after;
}

int bignumToInt(BignumType before)
  /*Returns NULL if bignum is too large*/
{
  unsigned int after;
  BIGDIGIT_TYPE *digitptr;
  
/* If bignum is too large return NULL and signal error*/
  if (BITS_IN(BIGDIGIT_TYPE) * before->numdigits > BITS_IN(int)) 
    ground_error(Func_Error,
		 "bignum to large to convert to int", NULL);
  digitptr = &(before->digits[before->numdigits - 1]);
  after = *digitptr;
  while (digitptr > before->digits) {
    digitptr --;
    after *= RADIX(BIGDIGIT_TYPE);
    after += *digitptr;
  }
  return after;
}

/**********************************/
/*Routines for printing bignums*/


/* structure used for printing bignums in base-10 */
/* digits are stored in an array with the ones digit first.*/
typedef struct PRINTME_STRUCT {
  unsigned long int numdigits;
  char digits[1];} *PrintmeType;


void printBignumStruct(BignumType bignum)
/* Debugging tool*/
/* outputs the structure of a bignum */
{
  unsigned long int digitCount = 0;

  switch (bignum->sign)
    {
    case -1: printf("\nnegative\n"); break;
    case 0: printf("\nzero\n"); break;
    case 1: printf("\npositive\n"); break;
    default: printf("\nwhaa?\n"); break;
    };
  printf("printBignumStruct: bignum has %i digits.\n", bignum->numdigits);
  for(; digitCount < bignum->numdigits ; digitCount++)
    printf("%i: %i ", digitCount, bignum->digits[digitCount]);
  printf("\n");
}


PrintmeType addIntToPrintme(PrintmeType printme, int addend)
{
  unsigned long int tempsum, carry=addend;
  char *digitptr, *endptr;

  digitptr = printme->digits;
  endptr = &(printme->digits[ printme->numdigits - 1 ]);
  while ((carry != 0) && (digitptr <= endptr)) {
    tempsum = carry + *digitptr;
    *digitptr = tempsum % 10;
    carry = tempsum / 10;
    digitptr ++;};
  while (carry != 0){
/*      printf("\naddIntToPrintme: reallocating\n");  */
      printme = 
	(PrintmeType) realloc(printme, (sizeof(struct PRINTME_STRUCT) + 
					(printme->numdigits)*(sizeof(char))));
      if (!printme) {
	ground_error(Func_Error, "addIntToPrintme: Allocation error", NULL);
      }
      printme->digits[ printme->numdigits ] = carry % 10;
      printme->numdigits++;
      carry = carry / 10;
    }
  return printme;
}			  
		  
PrintmeType multPrintmeByInt(PrintmeType printme, int factor)
{
  int carry=0;
  char *digitptr, *endptr;
  unsigned long int tempProd;

  digitptr = printme->digits;
  endptr = &( printme->digits[ printme->numdigits - 1 ] );
  while (digitptr <= endptr) {
    tempProd = factor * (*digitptr) + carry;
    *digitptr = tempProd % 10;
    carry = tempProd / 10;
    digitptr ++; }
  while (carry != 0) {
/*    printf("multPrintmeByInt: reallocating\n");    */
    printme = 
      (PrintmeType) realloc(printme, (sizeof(struct PRINTME_STRUCT) +
				      (printme->numdigits)*(sizeof(char))));
    if (!printme) {
      ground_error(Func_Error, "multPrintmeByInt: Allocation error", NULL);
    }
    printme->digits[ printme->numdigits ] = carry % 10;
    printme->numdigits ++;
    carry = carry / 10;
  }
  return printme;
}


char *BignumToString(Grounding gbig)
/*makes a structure of base-10 digits (1's digit 1st) & prints it backwards */
{
  BignumType bignum;
  PrintmeType printme;
  BIGDIGIT_TYPE *digitptr; char *printmeptr, *string, *strptr;
  char signflag = 0; bignum=GBIGNUM(gbig);
  if (bignum->sign==0) 
    {ALLOCATE(string,char,2); strcpy(string,"0"); return string;}
/* Initialize printme */
  ALLOCATE(printme,struct PRINTME_STRUCT,1);
  printme->numdigits=1;
  printme->digits[0]=0;
/* Make printme from bignum; digitptr starts at last digit of bignum*/
  digitptr = &( bignum->digits[ bignum->numdigits - 1 ] );
  printme = addIntToPrintme(printme, *digitptr); digitptr--;
  while(digitptr >= bignum->digits) {
    printme = multPrintmeByInt(printme, RADIX(BIGDIGIT_TYPE));
    printme = addIntToPrintme(printme, *digitptr);
    digitptr--; }
/* Prepare character string */
  string = fra_allocate(signflag + printme->numdigits + 1, sizeof(char));
  strptr = string;
  if (bignum->sign == -1) {
    signflag = 1;
    *string = '-';
    strptr ++;
  }
  FREE_BIGNUM(bignum);
  printmeptr = &(printme->digits[printme->numdigits - 1]);
  for (; printmeptr >= printme->digits; printmeptr--, strptr++) {
    *strptr = (char) (*printmeptr + '0');
  }
  *strptr = '\0';
  free(printme);
  return string;
}

/*****************************/
/*Routines for parsing bignums*/

unsigned long int intpow(unsigned long int base, int exp)
/*raises base to an integer power.  taken from Knuth*/
/*used by stringToBignum*/
{
  unsigned long int result = 1;
  char was_odd;

  if (!exp) return 1;
  while (1) {
    was_odd = exp & 1;
    exp >>= 1;
    if (was_odd) {
      result *= base;
      if (!exp) return result;
    }
    base *= base;
  }
}



unsigned long int charstogo(char *str)
/*Used to allow longer than standard strings
  maybe I don't even need it.*/
{
  unsigned long int togo = 0;

  while (*str) {
    togo ++;
    str ++;
  }
  return(togo);
}

unsigned long int readNextClump(char **stringptr, int *n)
  /*translates a clump of n digits from a string into an integer for
    stringToBignum*/
  /*Moves stringptr past the digits read, and changes n to let the 
     calling procedure know how many digits are left in the string*/
{
  int count;
  unsigned long int digitsleft, clumpval = 0;

  for (count=0; count < *n; count ++ ) {
    clumpval *= 10;
    clumpval += (*stringptr)[count] - '0';
  }
  *stringptr += *n;
  digitsleft = charstogo(*stringptr);
  if (*n > digitsleft) *n = digitsleft; 				    
  return clumpval;
}

Grounding stringToBignum(char *string)
/*Takes a base-10 number (as a string),
  and returns a grounding of a bignum with the same value as the string.*/
/*no check for non-digits*/
{
  Grounding ground; BignumType temp;
  int n=0; /*This variable will be set such that 10^n + a bigdigit will fit
	     an unsigned long int ...We'll take this many digits in a clump*/
  unsigned long int digitsleft;
  unsigned long int number;
  BignumType bignum = initializeBignum(__LINE__);
  number = ULONG_MAX;
  number >>= (BITS_IN(unsigned long) / 2);
/* number should be the largest integer such that (number*number) can be
   handled as a standard math operation without overflow */
  switch (string[0])
    {
    case '-':
      bignum->sign= (-1); string++;
      break;
    case '+':
      string++;
    default:
      bignum->sign= (1);
      break;
    }
  while ((number /= 10)) n++; /*this may be low*/
  /* printf("n = %i\n", n); */
  digitsleft = charstogo(string);
  if (n > digitsleft) (n = digitsleft);
  if (!bignum) {
    ground_error(Func_Error, "stringToBignum: Allocation error", NULL);
  }
  temp=bignum;
  bignum = addIntToBignum(bignum, readNextClump(&string, &n));
  if (temp != bignum) {FREE_BIGNUM(temp);}
  {
    BignumType temp;
    while (*string) {
      temp = bignum;
      bignum = multBignumByInt(bignum, intpow(10, n));
      if (bignum != temp) { FREE_BIGNUM(temp); temp = bignum; }
      bignum = addIntToBignum(bignum, readNextClump(&string, &n));
      if (bignum != temp) { FREE_BIGNUM(temp); temp = bignum; }
    }
  }
  if ((bignum->numdigits==1) && (bignum->digits[0]==0)) bignum->sign = 0;
  ground = bignum_to_ground(bignum);
  FREE_BIGNUM(bignum);
  return ground;
}

Grounding real_make_rational(Grounding numerator, Grounding denominator)
/*Automatically simplifies rationals and will change to integer if needed*/
{
  Grounding gcd, newnumer, newdenom, result;
  gcd = generic_gcd(numerator, denominator);
  newnumer = generic_div(numerator, gcd);
  newdenom = generic_div(denominator, gcd);
  if (generic_less_than(newdenom, a_zero)) {
    /* tempground = newdenom;  ????????????????????????????????should i free?
                                 it causes errors.*/
    
    newdenom = generic_times(newdenom, a_neg_one);

    /*FREE_GROUND(tempground);
    tempground = newnumer;
    */

    newnumer = generic_times(newnumer, a_neg_one);

    /*FREE_GROUND(tempground);
      */
  }
  if (generic_equal(newdenom, a_one)) {
    result = newnumer;
    FREE_GROUND(newdenom);
  }
  else {INITIALIZE_NEW_GROUND(result,rational_ground);
	(GCAR(result))=newnumer; USE_GROUND(newnumer);
	(GCDR(result))=newdenom; USE_GROUND(newdenom);}
  FREE_GROUND(gcd);
  return result;
}


extern char *(*big2string)();
extern Grounding (*string2big)();
extern Grounding (*make_real_rational)();

void init_fraxl_numerics()
{
  declare_lexpr(fraxl_plus_lexpr,"+"); declare_lexpr(fraxl_plus_lexpr,"plus");
  declare_lexpr(fraxl_minus_lexpr,"-"); declare_lexpr(fraxl_minus_lexpr,"minus");
  declare_lexpr(fraxl_times_lexpr,"*"); declare_lexpr(fraxl_times_lexpr,"times");
  declare_lexpr(fraxl_divide_lexpr,"/"); declare_lexpr(fraxl_divide_lexpr,"divide");
  declare_binary_function(generic_div,"div",any_ground,any_ground);
  declare_binary_function(generic_mod,"%",any_ground,any_ground);
  declare_binary_function(generic_mod,"mod",any_ground,any_ground);
  declare_binary_function(generic_greater_than,">",any_ground,any_ground);
  declare_binary_function(generic_less_than,"<",any_ground,any_ground);
  declare_binary_function(generic_equal,"=",any_ground,any_ground);
  declare_binary_function(generic_gcd,"gcd",any_ground,any_ground);
  declare_unary_function(fraxl_random,"random",integer_ground);
  a_zero=integer_to_ground(0);
  a_one=integer_to_ground(1);
  a_neg_one=integer_to_ground(-1);
  big2string=BignumToString;
  string2big=stringToBignum;
  make_real_rational=real_make_rational;
  reclaimers[(int) bignum_ground] = free_bignum;
}
