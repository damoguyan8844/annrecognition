#include <tpm/stream.h>

void main(int argc, char *argv[])
{
  FileHandle fp_in, fp_out;
  char fname[100];
  int i,length;
  double *x;

  if(argc < 2) {
    printf("Usage:\n%s outfile < input_files_file\n", argv[0]);
    exit(1);
  }
  
  fp_out = FileOpen(argv[1], "w");
  printf("Printing every 10th frame...\n");
  for(i=0;;i++) {
    getline(fname, 100, stdin);
    if(feof(stdin)) break;
    if(i % 10 == 0) printf("%s\n", fname);

    fp_in = FileOpen(fname, "r");
    
    fread(&length, sizeof(int), 1, fp_in);
    if(i == 0)
      fwrite(&length, sizeof(int), 1, fp_out);
    AdjustBuffer(&length, 1, sizeof(int));
    
    x = Allocate(length, double);
    fread(x, sizeof(double), length, fp_in);
    fwrite(x, sizeof(double), length, fp_out);
    free(x);
    FileClose(fp_in);
    unlink(fname);
  }

  FileClose(fp_out);
}
