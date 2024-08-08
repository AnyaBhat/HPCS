#include <stdio.h>
#include <omp.h>
void main()
{
	int sum=0;
	int N=1;
	int tid;
	omp_set_num_threads(N);
	#pragma omp parallel
	#pragma omp for
	for(int i=0;i<=20;i++)
	{
		tid=omp_get_thread_num();
		sum+=i;
		printf("sum =%d,thread %d \n",sum,tid);
	}
	
	printf("final sum=%d \n",sum);
}
