#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>

__global__ void swap ( int *ip, int len){
	int tid=threadIdx.x*2;
	if(tid <len-1){
		int temp=ip[tid];
		ip[tid]=ip[tid+1];
		ip[tid+1]=temp;
	}
}

int main(){
	int N = 6;  // Size of arrays
    	int size = N * sizeof(int);

    	int ip[N]={10,5,20,10,15,7};

    	int *d_ip;
	
	cudaMalloc((void **)&d_ip,size);
	
	cudaMemcpy(d_ip, ip, size, cudaMemcpyHostToDevice);
	printf("Orginal array: ");
	for(int i=0;i<N;i++){
		printf("%d ",ip[i]);
	}
	printf("\n");
	int n=N/2;
	swap<<<1,n>>>(d_ip,N);
	
	cudaMemcpy(ip,d_ip,size,cudaMemcpyDeviceToHost);
	printf("Swapped array:");
	for(int i=0;i<N;i++){
		printf("%d ",ip[i]);
	}
	printf("\n");
	cudaFree(d_ip);
	return 0;
}
