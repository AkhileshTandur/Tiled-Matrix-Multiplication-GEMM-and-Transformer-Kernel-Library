#include "kernels.hpp"
#include <chrono>
#include <iostream>
#include <cstdlib>
int main(int argc,char**argv){int n=argc>1?std::atoi(argv[1]):256,reps=argc>2?std::atoi(argv[2]):10;if(n<1||n>4096||reps<1)return 2;
std::vector<float>a(n*n,.25f),b(n*n,.5f),c(n*n);kernels::gemm(a.data(),b.data(),c.data(),n,n,n);
auto start=std::chrono::steady_clock::now();for(int r=0;r<reps;++r)kernels::gemm(a.data(),b.data(),c.data(),n,n,n);
double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/reps;
std::cout<<"{\"n\":"<<n<<",\"repetitions\":"<<reps<<",\"mean_seconds\":"<<sec<<",\"gflops\":"<<2.*n*n*n/sec/1e9<<",\"checksum\":"<<c[0]<<"}\n";}
