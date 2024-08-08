#include <stdio.h>
#include <omp.h>
void main(){
int N=20,i,sum=0;
omp_set_num_threads(4);
#pragma omp parallel for reduction(+:sum)
for(i=0;i<=20;i++){
	sum+=i;
}
	printf("Final sum=%d\n",sum);
}
