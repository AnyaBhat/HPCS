#include <stdio.h>
#include <omp.h>
void main()
{
	int a[5]={1,2,3,4,5};
	int b[5]={6,7,8,9,10};
	int c[5],tid;
	omp_set_num_threads(5);
	#pragma omp parallel 
	{
		tid=omp_get_thread_num();
		c[tid]=a[tid]+b[tid];
		printf("c[%d]=%d for thread %d \n",tid,c[tid],tid);
	}
}
