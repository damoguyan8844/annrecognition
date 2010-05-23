#include <tpm/tpm.h>

void main(int argc, char *argv[])
{
  FILE *fp_in, *fp_out;
  char *outfile;
  int length, threshold, i;
  double x;

  if(argc < 3) {
    printf("Usage:\n%s <infile> <outfile> [threshold]\n", argv[0]);
    exit(1);
  }
  
  fp_in = fopen(argv[1], "r");
  if(!fp_in) {
    printf("Cannot open input file `%s'\n", argv[1]);
    perror("");
    exit(1);
  }
  fp_out = fopen(argv[2], "wb");
  if(!fp_out) {
    printf("Cannot open output file `%s'\n", argv[2]);
    perror("");
    exit(1);
  }
  if(argc >= 4) threshold = atoi(argv[3]);
  else threshold = 0;
  
  for(length = 0; fscanf(fp_in, "%lf", &x) != EOF; length++);
  if(threshold && (threshold < length)) length = threshold;
  i = length;
  WriteAdjust(&i, TPM_INT, 1, fp_out);

  fseek(fp_in, 0L, SEEK_SET);
  for(i=0;fscanf(fp_in, "%lf", &x) != EOF;i++) {
    if(i == length) break;
    WriteAdjust(&x, TPM_DOUBLE, 1, fp_out);
  }

  fclose(fp_out);
  fclose(fp_in);
}
