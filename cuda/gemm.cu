#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cmath>
#include <cstdlib>
#define CHECK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(1);}}while(0)
constexpr int T=16;
__global__ void gemm(const float*a,const float*b,float*c,int m,int n,int k){
 __shared__ float aa[T][T],bb[T][T];int r=blockIdx.y*T+threadIdx.y,col=blockIdx.x*T+threadIdx.x;float sum=0;
 for(int t=0;t<k;t+=T){int ak=t+threadIdx.x,bk=t+threadIdx.y;
 aa[threadIdx.y][threadIdx.x]=(r<m&&ak<k)?a[r*k+ak]:0;
 bb[threadIdx.y][threadIdx.x]=(bk<k&&col<n)?b[bk*n+col]:0;__syncthreads();
 for(int z=0;z<T;++z)sum+=aa[threadIdx.y][z]*bb[z][threadIdx.x];__syncthreads();}
 if(r<m&&col<n)c[r*n+col]=sum;
}
int main(){int m=127,n=65,k=33;std::vector<float>a(m*k,.25f),b(k*n,.5f),c(m*n);float *da,*db,*dc;
 CHECK(cudaMalloc(&da,a.size()*4));CHECK(cudaMalloc(&db,b.size()*4));CHECK(cudaMalloc(&dc,c.size()*4));
 CHECK(cudaMemcpy(da,a.data(),a.size()*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(db,b.data(),b.size()*4,cudaMemcpyHostToDevice));
 dim3 blocks((n+T-1)/T,(m+T-1)/T),threads(T,T);gemm<<<blocks,threads>>>(da,db,dc,m,n,k);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
 CHECK(cudaMemcpy(c.data(),dc,c.size()*4,cudaMemcpyDeviceToHost));for(float v:c)if(std::abs(v-k*.125f)>1e-5)return 2;
 cudaEvent_t start,stop;CHECK(cudaEventCreate(&start));CHECK(cudaEventCreate(&stop));CHECK(cudaEventRecord(start));
 for(int i=0;i<200;++i)gemm<<<blocks,threads>>>(da,db,dc,m,n,k);CHECK(cudaEventRecord(stop));CHECK(cudaEventSynchronize(stop));float ms;CHECK(cudaEventElapsedTime(&ms,start,stop));
 printf("PASS; shape=%dx%dx%d mean_ms=%g GFLOP/s=%g\n",m,n,k,ms/200,2.*m*n*k/(ms/200*1e6));
 CHECK(cudaFree(da));CHECK(cudaFree(db));CHECK(cudaFree(dc));CHECK(cudaEventDestroy(start));CHECK(cudaEventDestroy(stop));}
