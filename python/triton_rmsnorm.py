"""Optional GPU RMSNorm. Run directly to validate and benchmark; no training."""
import torch
import triton
import triton.language as tl
@triton.jit
def _rms(X,W,Y,D:tl.constexpr,EPS:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0)
    col=tl.arange(0,B)
    x=tl.load(X+row*D+col,col<D,0).to(tl.float32)
    w=tl.load(W+col,col<D,0).to(tl.float32)
    inv=tl.rsqrt(tl.sum(x*x,axis=0)/D+EPS)
    tl.store(Y+row*D+col,x*inv*w,col<D)
def rmsnorm(x,w,eps=1e-5):
    if x.ndim!=2 or not x.is_cuda or not w.is_cuda or not x.is_contiguous() or not w.is_contiguous():
        raise ValueError('CUDA contiguous matrix and weight vector required')
    if w.shape!=(x.shape[1],) or w.device!=x.device or x.shape[1]<1 or x.shape[1]>8192 or eps<=0:
        raise ValueError('invalid shape, device, epsilon, or width >8192')
    y=torch.empty_like(x)
    _rms[(x.shape[0],)](x,w,y,x.shape[1],eps,triton.next_power_of_2(x.shape[1]))
    return y
if __name__=='__main__':
    torch.manual_seed(42)
    for dtype in [torch.float32,torch.float16,torch.bfloat16]:
        for d in [1,33,127,1024]:
            x=torch.randn(17,d,device='cuda',dtype=dtype);w=torch.randn(d,device='cuda',dtype=dtype)
            ref=(x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-5)*w.float()).to(dtype)
            torch.testing.assert_close(rmsnorm(x,w),ref,rtol=0.02 if dtype==torch.bfloat16 else 0.002,atol=0.02 if dtype==torch.bfloat16 else 0.002)
    x=torch.randn(1024,1024,device='cuda');w=torch.ones(1024,device='cuda')
    print({'device':torch.cuda.get_device_name(),'shape':list(x.shape),'median_ms':triton.testing.do_bench(lambda:rmsnorm(x,w))})
