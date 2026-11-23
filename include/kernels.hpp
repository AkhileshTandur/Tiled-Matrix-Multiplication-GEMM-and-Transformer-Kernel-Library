#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
namespace kernels {
inline void gemm(const float* a,const float* b,float* c,int m,int n,int k,int tile=32){
 if(m<=0||n<=0||k<=0||tile<=0)throw std::invalid_argument("positive dimensions required");
 std::fill(c,c+m*n,0.f);
 for(int ii=0;ii<m;ii+=tile)for(int kk=0;kk<k;kk+=tile)for(int jj=0;jj<n;jj+=tile)
 for(int i=ii;i<std::min(ii+tile,m);++i)for(int t=kk;t<std::min(kk+tile,k);++t)
 for(int j=jj;j<std::min(jj+tile,n);++j)c[i*n+j]+=a[i*k+t]*b[t*n+j];
}
inline void rmsnorm(const float* x,const float* weight,float* y,int rows,int cols,float eps=1e-5f){
 if(rows<=0||cols<=0||eps<=0)throw std::invalid_argument("invalid RMSNorm arguments");
 for(int r=0;r<rows;++r){double s=0;for(int j=0;j<cols;++j)s+=double(x[r*cols+j])*x[r*cols+j];
 double scale=1/std::sqrt(s/cols+eps);for(int j=0;j<cols;++j)y[r*cols+j]=float(x[r*cols+j]*scale*weight[j]);}
}
// Single-head, unbatched scaled dot-product attention. Online softmax avoids N*N storage.
inline void attention(const float* q,const float* k,const float* v,float* out,int n,int d,bool causal=false){
 if(n<=0||d<=0)throw std::invalid_argument("positive dimensions required");
 for(int i=0;i<n;++i){double maxscore=-INFINITY,denom=0;std::vector<double> acc(d,0);
 for(int j=0;j<(causal?i+1:n);++j){double score=0;for(int t=0;t<d;++t)score+=double(q[i*d+t])*k[j*d+t];score/=std::sqrt(double(d));
 double next=std::max(maxscore,score),alpha=std::exp(maxscore-next),beta=std::exp(score-next);
 for(int t=0;t<d;++t)acc[t]=alpha*acc[t]+beta*v[j*d+t];denom=alpha*denom+beta;maxscore=next;}
 for(int t=0;t<d;++t)out[i*d+t]=float(acc[t]/denom);}
}
}
