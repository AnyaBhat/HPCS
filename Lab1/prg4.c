#include <stdio.h>
#include <omp.h>
void main(){
int N=20,sum=0;
omp_set_num_threads(4);
#pragma omp parallel for collapse(2) reduction(+:sum)
for(int i=0;i<=N/10;i++){
	for(int j=1;j<=10;j++){
		sum+=(i-1)*10+j;
	}
}
	printf("Final sum=%d\n",sum);
}
