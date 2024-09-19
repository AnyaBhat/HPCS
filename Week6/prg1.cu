#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>

__global__ void addarr( float *a, float *b, float *c,int N){
	int tid=blockIdx.x*blockDim.x+threadIdx.x;
	if(tid<N){
	 	c[tid]=a[tid]+b[tid];
	}
}

int main(){
	int N = 5;  // Size of arrays
    	int size = N * sizeof(float);

    	float h_a[N]={1.0,2.0,3.0,4.0,5.0};
    	float h_b[N]={6.0,7.0,8.0,9.0,10.0};
    	float h_c[N];

    	float *d_a, *d_b, *d_c;
	
	cudaMalloc((void **)&d_a,size);
	cudaMalloc((void **)&d_b,size);
	cudaMalloc((void **)&d_c,size);
	
	cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
	
	addarr<<<N,1>>>(d_a,d_b,d_c,N);
	
	cudaMemcpy(h_c,d_c,size,cudaMemcpyDeviceToHost);
	
	for(int i=0;i<N;i++){
		printf("%f + %f = %f \n",h_a[i],h_b[i],h_c[i]);
	}
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	return 0;
}
