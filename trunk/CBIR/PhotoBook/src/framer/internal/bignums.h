/* C mode */

#define BIGDIGIT_TYPE unsigned char
/* must be unsigned and small enough to prevent overflow (at most half the
 size in memory of an unsigned long...but, there's a bug: unsigned short
 won't work right for long divisions*/
#define BITS_IN(type) (sizeof(type) * 8)
#define TWO_TO(n) ((unsigned long int) 1 << n)
#define RADIX(type) TWO_TO(BITS_IN(type))/*assumes unsigned!*/

/* Bignum math prototypes */

BignumType cleanUpBignum(BignumType bignum);

BignumType copyBignum(BignumType from);

char bigequal(BignumType num1, BignumType num2);

char biglessthan(BignumType num1, BignumType num2);

char biggreaterthan(BignumType num1, BignumType num2);

BignumType bigsum(BignumType addend1, BignumType addend2);

BignumType bigdiff(BignumType minuend, BignumType subtrahend);

BignumType bigprod(BignumType factor1, BignumType factor2);

BignumType bigdiv(BignumType numerator, BignumType denominator);

BignumType bigmod(BignumType numerator, BignumType denominator);

int intGCD(int x, int y);

BignumType bigGCD(BignumType u, BignumType v);

Grounding simplifyRational(Grounding rat);
Grounding FloatToRational(Grounding fl);
Grounding make_rational(Grounding num,Grounding denom);

BignumType signedIntToBignum(int before);

int bignumToInt(BignumType before);

/**************************************
Generic math routines
**************************************/


Grounding generic_times(Grounding x, Grounding y);

Grounding generic_divide(Grounding x, Grounding y);


Grounding bignum_to_ground(BignumType big);
