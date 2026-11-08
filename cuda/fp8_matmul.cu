// Requant Network research. Minimal cuBLASLt bridge; original MIT-licensed code.
#include <cublasLt.h>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>

static thread_local char error_message[256];
static void check(cublasStatus_t s) {
    if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS status " + std::to_string(int(s)));
}
struct Plan {
    cublasLtHandle_t handle{};
    cublasLtMatmulDesc_t operation{};
    cublasLtMatrixLayout_t a{}, b{}, c{}, d{};
    cublasLtMatmulPreference_t preference{};
    cublasLtMatmulHeuristicResult_t heuristic{};
    size_t workspace = 32 * 1024 * 1024;
    ~Plan() {
        if (preference) cublasLtMatmulPreferenceDestroy(preference);
        if (a) cublasLtMatrixLayoutDestroy(a);
        if (b) cublasLtMatrixLayoutDestroy(b);
        if (c) cublasLtMatrixLayoutDestroy(c);
        if (d) cublasLtMatrixLayoutDestroy(d);
        if (operation) cublasLtMatmulDescDestroy(operation);
        if (handle) cublasLtDestroy(handle);
    }
};
extern "C" const char* fp8_error() { return error_message; }
extern "C" void* fp8_create(int n, const float* scale, int fp8) {
    Plan* plan = new Plan;
    try {
        check(cublasLtCreate(&plan->handle));
        check(cublasLtMatmulDescCreate(&plan->operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        cublasOperation_t transa=CUBLAS_OP_T, transb=CUBLAS_OP_N;
        check(cublasLtMatmulDescSetAttribute(plan->operation,CUBLASLT_MATMUL_DESC_TRANSA,&transa,sizeof(transa)));
        check(cublasLtMatmulDescSetAttribute(plan->operation,CUBLASLT_MATMUL_DESC_TRANSB,&transb,sizeof(transb)));
        if (fp8 == 1) {
            check(cublasLtMatmulDescSetAttribute(plan->operation,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&scale,sizeof(scale)));
            check(cublasLtMatmulDescSetAttribute(plan->operation,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&scale,sizeof(scale)));
        }
        cudaDataType_t input = fp8 == 1 ? CUDA_R_8F_E4M3 : fp8 == 2 ? CUDA_R_16F : CUDA_R_32F;
        check(cublasLtMatrixLayoutCreate(&plan->a,input,n,n,n));
        check(cublasLtMatrixLayoutCreate(&plan->b,input,n,n,n));
        check(cublasLtMatrixLayoutCreate(&plan->c,CUDA_R_32F,n,n,n));
        check(cublasLtMatrixLayoutCreate(&plan->d,CUDA_R_32F,n,n,n));
        check(cublasLtMatmulPreferenceCreate(&plan->preference));
        check(cublasLtMatmulPreferenceSetAttribute(plan->preference,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&plan->workspace,sizeof(plan->workspace)));
        int count=0;
        check(cublasLtMatmulAlgoGetHeuristic(plan->handle,plan->operation,plan->a,plan->b,plan->c,plan->d,plan->preference,1,&plan->heuristic,&count));
        if (!count) throw std::runtime_error("No cuBLASLt heuristic for this layout");
        return plan;
    } catch (const std::exception& e) {
        std::snprintf(error_message,sizeof(error_message),"%s",e.what());delete plan;return nullptr;
    }
}
extern "C" int fp8_run(void* opaque,const void* a,const void* b,float* out,void* workspace,void* stream) {
    auto* p=static_cast<Plan*>(opaque);
    float alpha=1,beta=0;
    cublasStatus_t status=cublasLtMatmul(p->handle,p->operation,&alpha,a,p->a,b,p->b,&beta,out,p->c,out,p->d,
                                      &p->heuristic.algo,workspace,p->workspace,static_cast<cudaStream_t>(stream));
    return int(status);
}
extern "C" void fp8_destroy(void* opaque){delete static_cast<Plan*>(opaque);}
