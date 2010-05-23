/******************************************************************************
Copyright 1994 by the Massachusetts Institute of Technology.  All
rights reserved.

Developed by Thomas P. Minka and Rosalind W. Picard at the Media
Laboratory, MIT, Cambridge, Massachusetts, with support from BT, PLC,
Hewlett-Packard, and NEC.

Contributor to early versions of this software:
Fang Liu

This distribution is approved by Nicholas Negroponte, Director of
the Media Laboratory, MIT.

Permission to use, copy, or modify this software and its documentation
for educational and research purposes only and without fee is hereby
granted, provided that this copyright notice and the original authors'
names appear on all copies and supporting documentation.  If individual
files are separated from this distribution directory structure, this
copyright notice must be included.  For any other uses of this software,
in original or modified form, including but not limited to distribution
in whole or in part, specific prior permission must be obtained from
MIT.  These programs shall not be used, rewritten, or adapted as the
basis of a commercial software or hardware product without first
obtaining appropriate licenses from MIT.  MIT. makes no representations
about the suitability of this software for any purpose.  It is provided
"as is" without express or implied warranty.
******************************************************************************/

/******************************************************************************
This program implements the MRSAR feature extraction described in

J. Mao and A.K. Jain, 
"Texture classification and segmentation using 
multiresolution simultaneous autoregressive models", 
Pattern Recognition, vol. 25, no. 2, pp. 173-188, 1992.

It works by computing MRSAR features (with resolution levels 2,3,4) on
overlapping 21x21 subwindows of the image. It outputs the mean and inverse
covariance matrix of those feature vectors. Similarity between images can then
be computed using Mahalanobis distance, Gaussian divergence, etc.
******************************************************************************/

#include <tpm/tpm.h>
#include <tpm/vector.h>
#include <tpm/matrix.h>
#include <tpm/image.h>

#define MODEL_ORDER 2
#define LSE_SIZE (MODEL_ORDER*2)
#define NFEATURES (LSE_SIZE+1)
#define WINDOW_INC 2     /* windows will overlap by window_size - WINDOW_INC */

double sar_std(Matrix s, double *x);
Vector MatrixColSum(Matrix matrix);
Vector MatrixColMean(Matrix matrix);
double MatrixCorrMN(Matrix a, int m, int n);
void MatrixCorr(Matrix src, Matrix dest);
void lse_neighbor_sum(Matrix win, int d, Matrix sum);
Matrix MRSAR(Matrix image, int *res_levels, int num_levels, int window_size);

/* Dumps the feature vectors over the whole image to a file. */
void DumpFeatureImages(Matrix features)
{
  int i;
  int f;
  FILE *fp;

  fp = fopen("features", "w");
  for(i=0;i<features->rows;i++) {
    for(f=0;f<features->cols;f++) {
      fprintf(fp, "%g ", features->d2[i][f]);
    }
    fprintf(fp, "\n");
  }
  fclose(fp);
}

main(int argc, char *argv[])
{
  int res_levels[] = { 2,3,4 };
  int num_levels = 3;
  int window_size = 21;

  TpmImage im, img;
  Matrix image;
  Matrix features;
  Matrix cov;
  Vector mean;
  FILE *fp;
  int i,j;

  if(argc < 4) {
    printf("Usage: %s <infile> <coef_file> <cov_file> <icov_file>\n",
	   argv[0]);
    printf("To compute SAR coefficients from a PPM/PGM image.\n");
    printf("PPM images will be converted to gray before processing.\n");
    printf("The mean is written to <coef_file>.\n");
    printf("The covariance matrix is written to <cov_file>.\n");
    printf("The inverse covariance matrix is written to <icov_file>.\n");
    exit(0);
  }

  /* Read in the image as a double-precision matrix */
  im = TpmReadPnm(argv[1]);
  if(!im) exit(1);
  img = TpmImageGray(im);
  TpmImageFree(im);
  image = MatrixFromImage(img);
  TpmImageFree(img);

  /* Compute the matrix of features. */
  features = MRSAR(image, res_levels, num_levels, window_size);
  MatrixFree(image);

  /* Compute the covariance matrix by subtracting out the mean of each
   * coefficient and calculating the correlation.
   */
  mean = MatrixColMean(features);
  MatrixSubtractRow(features, mean->d);
  cov = MatrixCreate(features->cols, features->cols);
  MatrixCorr(features, cov);
  MatrixFree(features);

  /* Output the mean and covariance of the features */
  fp = fopen(argv[2], "w"); if(!fp) { perror("Cannot write mean"); exit(1); }
  VectorPrint(fp, mean);
  fclose(fp);
  VectorFree(mean);
  fp = fopen(argv[3], "w"); if(!fp) { perror("Cannot write covar"); exit(1); }
  MatrixPrint(fp, cov);
  fclose(fp);

  /* Invert the covariance, for use in the Mahalanobis distance */
  MatrixInvert(cov);
  fp = fopen(argv[4], "w"); if(!fp) { perror("Cannot write icovar"); exit(1); }
  MatrixPrint(fp, cov);
  fclose(fp);
  MatrixFree(cov);
/*
  DumpFeatureImages(features);
*/
  MEM_BLOCKS();
  MEM_STATUS();
  exit(0);
}

/* Computes the SAR parameters on the image by iterating
 * over overlapping windows of size <window_size> and 
 * estimating the AR model at each of the specified resolution levels.
 * The output matrix is a set of row vectors, where each row is
 * the estimated AR parameters (4 numbers) and the std dev of the error
 * (1 number) for each resolution level. For example, for the three
 * resolution levels 2,3,4, each row will have 15 numbers, 5 for each level:
 * | AR 2 | dev 2 | AR 3 | dev 3 | AR 4 | dev 4 |
 */
Matrix MRSAR(Matrix image, int *res_levels, int num_levels, int window_size)
{
  Matrix window;                 /* work window */
  double *p;
  int win_y, win_x, i,j,k, lvl;
  Matrix sum,                    /* work matrix for estimation */
         A;                      /* correlation of the sum matrix */
  double *B;                     /* estimated parameters */
  Matrix features;               /* output array */
  int eff_width, eff_height;     /* dimensions of the useful image */

  /* ignore the left and bottom borders of the image;
     round to WINDOW_INC. */
  eff_height = image->rows - window_size + 1;
  eff_height = eff_height / WINDOW_INC * WINDOW_INC;
  eff_width = image->cols - window_size + 1;
  eff_width = eff_width / WINDOW_INC * WINDOW_INC;

  /* create the matrices */
  features = MatrixCreate(eff_height*eff_width, NFEATURES * num_levels);
  window = MatrixCreate(window_size, window_size);
  sum = MatrixCreate(window_size*window_size, NFEATURES);
  A = MatrixCreate(LSE_SIZE, LSE_SIZE);
  B = Allocate(NFEATURES, double);

  /* loop across windows of the image */
  for(win_y = 0; win_y < eff_height; win_y += WINDOW_INC) {
    fprintf(stderr, "Row: %d\n", win_y);
    for(win_x = 0; win_x < eff_width; win_x += WINDOW_INC) {

      /* copy image data into the window */
      p = window->d;
      for(i = 0; i < window->rows; i++)
	for(j = 0; j < window->cols; j++)
	  *p++ = image->d2[win_y+i][win_x+j];

      /* debias the window */
      VectorAddValue((Vector)window, -VectorSum((Vector)window)/window->len);

      /* loop resolution levels */
      for(lvl = 0; lvl < num_levels; lvl++) {

	/* compute the features over the window for this resolution */
	/* sum[i][LSE_SIZE] is the pixel value itself */
	lse_neighbor_sum(window, res_levels[lvl], sum);

	/* let A be the correlation matrix */
	MatrixCorr(sum, A);

	/* let B be the correlations with the pixel value */
	for(i=0; i<LSE_SIZE; i++) {
	  B[i] = MatrixCorrMN(sum, i, LSE_SIZE);
	}

	/* estimate the parameters; place in B */
	MatrixSolve(A, B);
	/* let the last element be the std dev of the error */
	B[LSE_SIZE] = sar_std(sum, B);

	/* put into the feature array */
	for(i=0;i<WINDOW_INC;i++) {
	  for(j=0;j<WINDOW_INC;j++) {
	    k = (win_y+i)*eff_width + (win_x+j);
	    memcpy(&features->d2[k][lvl*NFEATURES],
		   B, NFEATURES*sizeof(double));
	  }
	}
      }
    }
  }
  MatrixFree(window);
  MatrixFree(sum);
  MatrixFree(A);
  free(B);
  return features;
}

/* Vector dot product. */
double dot_product(double *a, double *b, int len)
{
  double r = 0.0;
  while(len--) r += *a++ * *b++;
  return r;
}

/* Compute the std dev of the error from the estimated parameters
 * to the true pixel values.
 */
double sar_std(Matrix s, double *x)
{
  int i;
  double sum = 0.0, v;

  for(i=0;i<s->rows;i++) {
    v = dot_product(s->d2[i], x, LSE_SIZE) - s->d2[i][LSE_SIZE];
    sum += v*v;
  }
  v = sqrt(sum / s->rows);
  return v;
}

/* Compute a row vector of column sums */
Vector MatrixColSum(Matrix matrix)
{
  Vector row;
  int i,j;

  row = VectorCreate(matrix->cols);
  VectorSet(row, 0.0);
  for(i=0;i<matrix->rows;i++) {
    for(j=0;j<matrix->cols;j++) {
      row->d[j] += matrix->d2[i][j];
    }
  }
  return row;
}

/* Compute a row vector of column means */
Vector MatrixColMean(Matrix matrix)
{
  Vector row;

  row = MatrixColSum(matrix);
  VectorScale(row, 1.0/matrix->rows);
  return row;
}

/* Computes the correlation between columns m and n in a */
double MatrixCorrMN(Matrix a, int m, int n)
{
  int i;
  double r = 0.0;

  for(i=0;i<a->rows;i++) {
    r += a->d2[i][m] * a->d2[i][n];
  }
  return r / a->rows;
}

/* Computes a matrix of correlations of src, placing it in dest */
void MatrixCorr(Matrix src, Matrix dest)
{
  int i,j;

  for(i=0;i<dest->rows;i++) {
    for(j=i;j<dest->cols;j++) {
      dest->d2[j][i] = dest->d2[i][j] = MatrixCorrMN(src, i, j);
    }
  }
}

/* Computes symmetric, noncausal AR features over all points in win */
void lse_neighbor_sum(Matrix win, int d, Matrix sum)
{
  int i,j, i1,i2, j1,j2;
  double *p;

  p = sum->d2[0];
  for(i=0;i<win->rows;i++) {
    i1 = (i - d + win->rows) % win->rows;
    i2 = (i + d) % win->rows;
    for(j=0;j<win->cols;j++) {
      j1 = (j - d + win->cols) % win->cols;
      j2 = (j + d) % win->cols;
      *p++ = win->d2[i1][j] + win->d2[i2][j];
      *p++ = win->d2[i][j1] + win->d2[i][j2];
      *p++ = win->d2[i1][j1] + win->d2[i2][j2];
      *p++ = win->d2[i1][j2] + win->d2[i2][j1];
      *p++ = win->d2[i][j];    /* last entry is true pixel value */
    }
  }
}
