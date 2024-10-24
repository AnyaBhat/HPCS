#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

__global__ void k_cal(int *d_a,int *d_b,int N){
  int tid=blockIdx.x*blockDim.x+threadIdx.x;
  if(tid<N*N){
    int i=tid/N;
    int j=tid%N;
    if(i+j==N-1){
      d_b[i*N+j]=0;
    }
    else if(i<N-1 && j<N-i-1){
      d_b[i*N+j]=d_a[i*N+j]*2;
    }else{
      d_b[i*N+j]=d_a[i*N+j]*d_a[i*N+j]*d_a[i*N+j];
    }
  }
}

int main(){
  int N;
  printf("Enter the width of matrix:");
  scanf("%d",&N);
  int *h_a,*h_b;
  size_t size=N*N*sizeof(int);
  h_a=(int *)malloc(size);
  h_b=(int *)malloc(size);
  int *d_a,*d_b;
  cudaMalloc(&d_a,size);
  cudaMalloc(&d_b,size);
  printf("Enter the elements:\n");
  for(int i=0;i<N*N;i++){
    scanf("%d",&h_a[i]);
  }
  cudaMemcpy(d_a,h_a,size,cudaMemcpyHostToDevice);
  k_cal<<<1,N*N>>>(d_a,d_b,N);
  cudaMemcpy(h_b,d_b,size,cudaMemcpyDeviceToHost);
  printf("The output is:\n");
  for(int i=0;i<N*N;i++){
      printf("%d ",h_b[i]);
      if((i+1)%N==0){
        printf("\n");
      }
  }
  cudaFree(d_a);
  cudaFree(d_b);
  free(h_a);
  free(h_b);
  return 0;
}

/* 
nvcc -o final_exam final_exam.cu
./final_exam
Enter the width of matrix:4
Enter the elements:
1 2 3 4 
5 6 7 8
1 2 3 4
3 4 5 6
The output is:
2 4 6 0 
10 12 0 512 
2 0 27 64 
0 64 125 216 
*/
