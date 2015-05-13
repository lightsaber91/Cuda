#include <stdio.h>
#include <assert.h>
#define epsilon (float)1e-5
#define DATA double
#define THREADxBLOCKalongXorY 4

void MatrixMulOnHost(DATA* M, DATA* N, DATA* P, int Width) {
    for (int i = 0; i < Width; ++i) {
        for (int j = 0; j < Width; ++j) {
            double pvalue = 0;
            for (int k = 0; k < Width; ++k) {
                double a = M[i * Width + k];
                double b = N[k * Width + j];
                pvalue += a * b;
            }
        P[i * Width + j] = pvalue;
        }
    }
}

__global__ void MatrixMulKernel(DATA* dM, DATA* dN, DATA* dP, int Width) {
    // Pvalue  utilizzato per il calcolo dell elemento di matrice
    // assegnato al thread
   DATA Pvalue = 0.0;

   int ix   = blockIdx.x*blockDim.x + threadIdx.x;
   int iy   = blockIdx.y*blockDim.y + threadIdx.y;

   int idx=iy*Width+ix;
   if(ix<Width && iy<Width) {
      for (int k = 0; k < Width; ++k) {
         DATA Melement = dM[iy*Width+k];
         DATA Nelement = dN[k*Width+ix];
         Pvalue += Melement * Nelement;
      }
      dP[idx] = Pvalue;
   }
}


void MatrixMulOnDevice(DATA* M, DATA* N, DATA* P, int Width, float *et) {

    int size = Width * Width * sizeof(DATA);

    cudaEvent_t start, stop;


    DATA *dM, *dN, *dP;
    int gridside = Width/THREADxBLOCKalongXorY;
    if(gridside*THREADxBLOCKalongXorY < Width) {
       gridside=gridside+1;
    }

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

  // Allocazione  e caricamento di M ed N sulla memoria GPU
    cudaMalloc(&dM, size);
    cudaMemcpy(dM, M, size, cudaMemcpyHostToDevice);

    cudaMalloc(&dN, size);
    cudaMemcpy(dN, N, size, cudaMemcpyHostToDevice);

    cudaMalloc(&dP, size);
  // Setup the execution configuration

    dim3 dimGrid(gridside, gridside);
    dim3 dimBlock(THREADxBLOCKalongXorY, THREADxBLOCKalongXorY);

    cudaEventRecord(start, 0);



    // Lancio dei thread per l esecuzione del kernel!
    printf("Num blocchi: %d -- Num Thread: %d\n", dimGrid.x, dimBlock.x);
    MatrixMulKernel<<<dimGrid, dimBlock>>>(dM, dN, dP, Width);


    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);


    cudaEventElapsedTime(et, start, stop);


  // Copia P dalla memoria GPU
    cudaMemcpy(P, dP, size, cudaMemcpyDeviceToHost);

  // Libera la memoria utilizzata per le matrici
    cudaFree(dM);
    cudaFree(dN);
    cudaFree(dP);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

}


// main

int main(int argc, char** argv) {

  int Width;
  float et;

  DATA *M, *N, *hP, *gP;

  if(argc<2) {
     fprintf(stderr,"Usage: %s Width\n",argv[0]);
     exit(1);
  }

  Width=atoi(argv[1]);

  if(Width<1) {
     fprintf(stderr,"Error Width=%d, must be > 0\n",Width);
     exit(1);

  }

  M=(DATA *)malloc(Width*Width*sizeof(DATA));
  N=(DATA *)malloc(Width*Width*sizeof(DATA));
  hP=(DATA *)malloc(Width*Width*sizeof(DATA));
  gP=(DATA *)malloc(Width*Width*sizeof(DATA));

  if(M==NULL) {
    fprintf(stderr,"Could not get memory for M\n");
    exit(1);
  }
  if(N==NULL) {
    fprintf(stderr,"Could not get memory for N\n");
    exit(1);
  }
  if(hP==NULL) {
    fprintf(stderr,"Could not get memory for hP\n");
    exit(1);
  }
  if(gP==NULL) {
    fprintf(stderr,"Could not get memory for gP\n");
    exit(1);
  }

  memset(gP,0,Width*Width*sizeof(DATA));
  memset(hP,0,Width*Width*sizeof(DATA));

  for(int y=0; y<Width; y++){
    printf("\n");
    for(int x=0; x<Width; x++) {
       M[y*Width+x]=(DATA)(y*Width+x);
       N[y*Width+x]=(DATA)(y*Width+x);
    }
  }

  MatrixMulOnHost(M, N, hP, Width);
  MatrixMulOnDevice(M, N, gP, Width, &et);

  printf("\n\nInput Matrix");
  for(int y=0; y<Width; y++){
    printf("\n");
    for(int x=0; x<Width; x++) {
       printf("%d ", M[y*Width+x]);
    }
  }


  printf("\n\nplain C");
  for(int y=0; y<Width; y++){
    printf("\n");
    for(int x=0; x<Width; x++) {
       printf("%d ", hP[y*Width+x]);
    }
  }

  printf("\n\nGPU C");
  for(int y=0; y<Width; y++){
    printf("\n");
    for(int x=0; x<Width; x++) {
       printf("%d ", gP[y*Width+x]);
    }
  }

  int errCnt = 0;
  for(int y=0; y<Width; y++){
   for(int x=0; x<Width; x++) {
      DATA it = hP[y*Width+x];
      if(fabs(it - gP[y*Width+x])> epsilon*it)
       errCnt++;

   }
  }

  if(errCnt==0) {
    printf("\nTEST PASSED\n");
    printf("Kernel execution time=%f milliseconds\n",et);
  } else {
    printf("\n\nTEST FAILED: number of errors:  %d\n", errCnt);
  }

}
