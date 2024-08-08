#include <stdio.h>
#include <omp.h>
void main()
{
	omp_set_num_threads(4);
	int sum=0;
	int N=20;
	int tid;
	#pragma omp parallel
	#pragma omp for reduction(+:sum)
	{
		tid=omp_get_thread_num();
		sum+=N;
		printf("sum =%d, thread %d \n",sum,tid);

	}
	
	printf("final sum=%d \n",sum);
}
