#include <stdio.h>
#include <omp.h>
void main()
{
	omp_set_num_threads(4);
	int sum=0;
	int N=20;
	int tid;
	#pragma omp parallel
	#pragma omp for
	for(int i=0;i<4;i++)
	{
		int localsum=0;
		tid=omp_get_thread_num();
		localsum+=N;
		#pragma omp critical
		{
			sum+=N;
			printf("sum =%d, localsum=%d, thread %d \n",sum,localsum,tid);
		}
	}
	
	printf("final sum=%d \n",sum);
}
