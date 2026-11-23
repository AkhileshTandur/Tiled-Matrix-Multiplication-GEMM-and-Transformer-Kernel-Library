#include "kernels.hpp"
#include <iostream>
#include <random>
#include <chrono>
#include <cstdlib>
void near(double a,double b){if(std::abs(a-b)>1e-4*(1+std::abs(b)))throw std::runtime_error("numerical mismatch");}
int main(){
 std::mt19937 rng(42);std::uniform_real_distribution<float> dist(-1,1);
 for(int m:{1,7,33})for(int n:{1,9,35})for(int k:{1,5,37}){
 std::vector<float>a(m*k),b(k*n),c(m*n);for(auto&x:a)x=dist(rng);for(auto&x:b)x=dist(rng);
 kernels::gemm(a.data(),b.data(),c.data(),m,n,k);
 for(int i=0;i<m;++i)for(int j=0;j<n;++j){double ref=0;for(int t=0;t<k;++t)ref+=double(a[i*k+t])*b[t*n+j];near(c[i*n+j],ref);}}
 for(int n:{1,7,17})for(int d:{1,5,33}){
 std::vector<float>q(n*d),k(n*d),v(n*d),o(n*d),y(n*d),w(d,1);
 for(auto*x:{&q,&k,&v})for(auto&t:*x)t=dist(rng);
 kernels::rmsnorm(q.data(),w.data(),y.data(),n,d);
 for(int i=0;i<n;++i){double ss=0;for(int t=0;t<d;++t)ss+=double(q[i*d+t])*q[i*d+t];for(int t=0;t<d;++t)near(y[i*d+t],q[i*d+t]/std::sqrt(ss/d+1e-5));}
 for(bool causal:{false,true}){kernels::attention(q.data(),k.data(),v.data(),o.data(),n,d,causal);
 for(int i=0;i<n;++i){int count=causal?i+1:n;std::vector<double>s(count);double mx=-INFINITY;
 for(int j=0;j<count;++j){s[j]=0;for(int t=0;t<d;++t)s[j]+=double(q[i*d+t])*k[j*d+t];s[j]/=std::sqrt(double(d));mx=std::max(mx,s[j]);}
 double den=0;for(auto&t:s){t=std::exp(t-mx);den+=t;}
 for(int t=0;t<d;++t){double ref=0;for(int j=0;j<count;++j)ref+=s[j]*v[j*d+t]/den;near(o[i*d+t],ref);}}}}
 bool rejected=false;try{float x=0;kernels::gemm(&x,&x,&x,0,1,1);}catch(const std::invalid_argument&){rejected=true;}if(!rejected)return 2;
 std::cout<<"PASS: GEMM irregular dimensions, RMSNorm, causal/noncausal attention, invalid dimensions\n";
}
