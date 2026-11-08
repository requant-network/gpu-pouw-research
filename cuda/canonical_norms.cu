// Requant Network research. Independent exact integer-frame norm implementation.
#include <cuda_runtime.h>
#include <cmath>
using U=unsigned int;using W=unsigned long long;
__device__ U bf16(float x){U u=__float_as_uint(x);return (u+0x7fff+((u>>16)&1))>>16;}
__device__ int compare_shift(W s,int d,W bound){
    if(s==0)return -1;
    int a=64-__clzll(s),b=64-__clzll(bound);
    if(d>=0){if(a+d!=b)return a+d>b?1:-1;s<<=d;}
    else{if(a!=b-d)return a>b-d?1:-1;bound<<=-d;}
    return s>bound?1:s<bound?-1:0;
}
__device__ U sqrt_code(W sum,U emax,U width,U k){
    if(!sum)return 0;
    double value=ldexp(double(sum)/double(k),int(emax)-268-int(width));
    U code=bf16(float(sqrt(value)));
    for(int iteration=0;iteration<8;iteration++){
        U exp=code>>7,t=(code&127)+(exp?128:0),f=exp?exp:1;
        W upper=W(4*t+2)*(4*t+2)*k;
        int d=int(emax)+4-int(width)-2*int(f),hi=compare_shift(sum,d,upper);
        bool hi_ok=hi<0||(hi==0&&!(t&1)),lo_ok=true;
        if(t){W low=4*t-2+(t==128&&f>=2);int lo=compare_shift(sum,d,low*low*k);lo_ok=lo>0||(lo==0&&!(t&1));}
        if(lo_ok&&hi_ok)return code;
        code=lo_ok?code+1:code-1;
    }
    return 0xffff;
}
__global__ void norms(const signed char* codes,const unsigned short* scales,unsigned short* out,int k){
    __shared__ U exponents[256];__shared__ float maxima[256];__shared__ W sums[256];
    int row=blockIdx.x,t=threadIdx.x,groups=k/8;U maximum=0;float largest=0;
    for(int g=t;g<groups;g+=256){
        U bits=scales[row*groups+g]&0x7fff,exp=bits>>7,m=(bits&127)+(exp?128:0),amax=0;
        for(int i=0;i<8;i++){int v=codes[row*k+8*g+i];U a=v<0?-v:v;amax=amax>a?amax:a;}
        if(amax&&m)maximum=maximum>2*(exp?exp:1)?maximum:2*(exp?exp:1);
        largest=fmaxf(largest,amax*__uint_as_float(bits<<16));
    }
    exponents[t]=maximum;maxima[t]=largest;__syncthreads();
    for(int step=128;step;step>>=1){if(t<step){exponents[t]=max(exponents[t],exponents[t+step]);maxima[t]=fmaxf(maxima[t],maxima[t+step]);}__syncthreads();}
    U log=0;while((1U<<log)<U(k))log++;U width=27-log,emax=exponents[0];W total=0;
    for(int g=t;g<groups;g+=256){
        U bits=scales[row*groups+g]&0x7fff,exp=bits>>7,m=(bits&127)+(exp?128:0),q2=0;
        for(int i=0;i<8;i++){int v=codes[row*k+8*g+i];q2+=v*v;}
        W product=W(m)*m*q2;int shift=int(emax)-2*int(exp?exp:1);
        if(product&&shift>=0&&shift<64)total+=(product<<width)>>shift;
    }
    sums[t]=total;__syncthreads();
    for(int step=128;step;step>>=1){if(t<step)sums[t]+=sums[t+step];__syncthreads();}
    if(t==0){U code=sqrt_code(sums[0],emax,width,k);code=(code+2)&0xfffc;
        out[2*row]=code<0x7f80?code:0xffff;out[2*row+1]=bf16(maxima[0]);}
}
// Restricted experiment adapter: encode known canonical BF16 norms as synthetic
// float stats for the unmodified scale chain. Validated only on finite test rows;
// production integration should pass canonical norm codes directly (no FP32 square).
__global__ void inject(const unsigned short* norm,float* stats,int k){
    int row=blockIdx.x,parts=k/512;
    for(int i=threadIdx.x;i<parts*2;i+=256)stats[row*parts*2+i]=0;
    __syncthreads();
    if(threadIdx.x==0){float l2=__uint_as_float(U(norm[2*row])<<16),linf=__uint_as_float(U(norm[2*row+1])<<16);
        stats[row*parts*2]=l2*l2*float(k);stats[row*parts*2+1]=linf;}
}
extern "C" int requant_norms(const signed char* codes,const unsigned short* scales,unsigned short* out,int rows,int k,void* stream){
    if(rows<=0||k<1024||k>65536||k%512)return int(cudaErrorInvalidValue);
    norms<<<rows,256,0,static_cast<cudaStream_t>(stream)>>>(codes,scales,out,k);
    return int(cudaGetLastError());
}
extern "C" int requant_inject(const unsigned short* norms,float* stats,int rows,int k,void* stream){
    if(rows<=0||k<1024||k>65536||k%512)return int(cudaErrorInvalidValue);
    inject<<<rows,256,0,static_cast<cudaStream_t>(stream)>>>(norms,stats,k);
    return int(cudaGetLastError());
}
