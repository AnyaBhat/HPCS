#include<cuda_runtime.h>
#include<cuda.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>

__global__ void toggle(char *str,char *tg_str,int len){
	int idx=threadIdx.x;
	if(idx<len){
		char ch=str[idx];
		if(ch>='A' && ch<='Z'){
			tg_str[idx]=(ch+32);
		}else if(ch>='a'&&ch<='z'){
			tg_str[idx]=(ch-32);
		}else{
			tg_str[idx]=ch;
		}
	}
}

int main(){
	char str[256];
	char *gpu_str,*tg_str;
	
	printf("Enter the string: ");
	scanf("%255s",str);
	
	int len=strlen(str);
	char res_str[len];
	int size=len*sizeof(char);
	
	cudaMalloc((void**)&gpu_str,size);
	cudaMalloc((void**)&tg_str,size);
	
	cudaMemcpy(gpu_str,str,size,cudaMemcpyHostToDevice);
	
	toggle<<< 1,len>>>(gpu_str,tg_str,len);
	
	cudaMemcpy(res_str,tg_str,size,cudaMemcpyDeviceToHost);
	
	res_str[len]='\0';
	
	printf("Final toggled string=%s\n",res_str);
	
	return 0;
}
