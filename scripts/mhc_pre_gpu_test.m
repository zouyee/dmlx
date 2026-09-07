// Standalone operational-candidate microbenchmark for production mhc_pre_split_weighted_sum_norm.
// Diagnostic only: no test-only kernel is referenced by production code.
// Build/run: clang -framework Metal -framework Foundation -fobjc-arc \
//   scripts/mhc_pre_gpu_test.m -o /tmp/mhc_pre_gpu_test && /tmp/mhc_pre_gpu_test
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <math.h>
#import <stdint.h>
#import <stdio.h>
#import <stdlib.h>
#import <string.h>
#import <unistd.h>

#define DIM 4096u
#define HC 4u
#define MHC_H (HC * DIM)
#define MIX3 24u
#define WARMUP 10
#define REPS 21

static const char *variants =
"kernel void mhc_pre_diag_dot_only(device const float* fn [[buffer(0)]], device const float* residual [[buffer(1)]], constant float& norm [[buffer(2)]], device float* out [[buffer(3)]], uint lid [[thread_position_in_threadgroup]], uint n [[threads_per_threadgroup]]) {"
" for(uint r=lid;r<24;r+=n){float a=0;device const float* w=fn+r*16384;for(uint i=0;i<16384;i++)a+=w[i]*residual[i];out[r]=a*norm;} }"
"kernel void mhc_pre_diag_phase_b_only(device const float* residual [[buffer(0)]], device const float* pre [[buffer(1)]], device const float* nw [[buffer(2)]], device float* ai [[buffer(3)]], device bfloat* no [[buffer(4)]], uint lid [[thread_position_in_threadgroup]], uint n [[threads_per_threadgroup]]) {"
" threadgroup float row[4096]; threadgroup float ss[256]; float z=0;for(uint d=lid;d<4096;d+=n){float a=0;for(uint m=0;m<4;m++)a+=pre[m]*residual[m*4096+d];float v=float(bfloat(a));row[d]=v;z+=v*v;}"
" ss[lid]=z;threadgroup_barrier(mem_flags::mem_threadgroup);for(uint s=n/2;s;s>>=1){if(lid<s)ss[lid]+=ss[lid+s];threadgroup_barrier(mem_flags::mem_threadgroup);}float k=rsqrt(ss[0]/4096.0f+1e-6f);for(uint d=lid;d<4096;d+=n){ai[d]=row[d];no[d]=bfloat(row[d]*k*nw[d]);} }";

static uint32_t rng_state = 0x12345678u;
static float nextf(void) { rng_state = rng_state * 1664525u + 1013904223u; return ((rng_state >> 8) / 16777216.0f - 0.5f) * 0.1f; }
static int cmp_u64(const void *a,const void *b){uint64_t x=*(const uint64_t*)a,y=*(const uint64_t*)b;return x>y?1:x<y?-1:0;}
static uint16_t f32_bf16(float x){uint32_t u;memcpy(&u,&x,4);u+=0x7fffu+((u>>16)&1u);return (uint16_t)(u>>16);}
static float bf16_f32(uint16_t x){uint32_t u=(uint32_t)x<<16;float v;memcpy(&v,&u,4);return v;}
static bool check_reference(id<MTLBuffer> __strong const *b,float residual_norm){
    const float *fw=b[0].contents,*base=b[1].contents,*scale=b[2].contents,*res=b[3].contents,*nw=b[7].contents;
    const float *gpu_post=b[4].contents,*gpu_comb=b[5].contents,*gpu_ai=b[6].contents,*gpu_dot=b[9].contents;const uint16_t *gpu_no=b[8].contents;
    float mix[24],pre[4],post[4],comb[16],row[4096];
    for(uint r=0;r<24;r++){float a=0;for(uint i=0;i<16384;i++)a+=fw[r*16384+i]*res[i];mix[r]=a*residual_norm;if(fabsf(gpu_dot[r]-mix[r])>0.05f)return false;}
    for(uint i=0;i<4;i++){pre[i]=1.0f/(1.0f+expf(-(mix[i]*scale[0]+base[i])))+1e-6f;post[i]=2.0f/(1.0f+expf(-(mix[4+i]*scale[1]+base[4+i])));if(fabsf(gpu_post[i]-post[i])>1e-4f)return false;}
    for(uint i=0;i<16;i++)comb[i]=mix[8+i]*scale[2]+base[8+i];
    for(uint i=0;i<4;i++){float m=comb[i*4];for(uint j=1;j<4;j++)if(comb[i*4+j]>m)m=comb[i*4+j];float s=0;for(uint j=0;j<4;j++){comb[i*4+j]=expf(comb[i*4+j]-m);s+=comb[i*4+j];}for(uint j=0;j<4;j++)comb[i*4+j]=comb[i*4+j]/s+1e-6f;}
    for(uint j=0;j<4;j++){float s=1e-6f;for(uint i=0;i<4;i++)s+=comb[i*4+j];for(uint i=0;i<4;i++)comb[i*4+j]/=s;}
    for(uint it=0;it<19;it++){for(uint i=0;i<4;i++){float s=1e-6f;for(uint j=0;j<4;j++)s+=comb[i*4+j];for(uint j=0;j<4;j++)comb[i*4+j]/=s;}for(uint j=0;j<4;j++){float s=1e-6f;for(uint i=0;i<4;i++)s+=comb[i*4+j];for(uint i=0;i<4;i++)comb[i*4+j]/=s;}}
    for(uint i=0;i<16;i++)if(fabsf(gpu_comb[i]-comb[i])>1e-4f)return false;
    double ss=0;for(uint d=0;d<4096;d++){float a=0;for(uint m=0;m<4;m++)a+=pre[m]*res[m*4096+d];row[d]=bf16_f32(f32_bf16(a));if(fabsf(gpu_ai[d]-row[d])>1e-6f)return false;ss+=(double)row[d]*row[d];}
    float k=1.0f/sqrtf((float)(ss/4096.0)+1e-6f);for(uint d=0;d<4096;d++){float expected=bf16_f32(f32_bf16(row[d]*k*nw[d]));if(fabsf(bf16_f32(gpu_no[d])-expected)>0.02f)return false;}
    return true;
}

static id<MTLComputePipelineState> pipeline(id<MTLDevice> d,id<MTLLibrary> l,NSString *name){NSError *e=nil;id<MTLFunction> f=[l newFunctionWithName:name];id<MTLComputePipelineState> p=f?[d newComputePipelineStateWithFunction:f error:&e]:nil;if(!p){fprintf(stderr,"pipeline %s failed: %s\n",name.UTF8String,e.localizedDescription.UTF8String);exit(1);}return p;}

static uint64_t run_one(id<MTLDevice> d,id<MTLCommandQueue> q,id<MTLComputePipelineState> p,id<MTLCounterSet> cs,int kind,float residual_norm,id<MTLBuffer> __strong const *b){
    MTLCounterSampleBufferDescriptor *sd=[MTLCounterSampleBufferDescriptor new];sd.counterSet=cs;sd.sampleCount=2;sd.storageMode=MTLStorageModeShared;NSError *err=nil;id<MTLCounterSampleBuffer> sb=[d newCounterSampleBufferWithDescriptor:sd error:&err];if(!sb){fprintf(stderr,"sample buffer failed: %s\n",err.localizedDescription.UTF8String);exit(1);}
    MTLComputePassDescriptor *pd=[MTLComputePassDescriptor computePassDescriptor];id a=pd.sampleBufferAttachments[0];[a setSampleBuffer:sb];[a setStartOfEncoderSampleIndex:0];[a setEndOfEncoderSampleIndex:1];
    id<MTLCommandBuffer> cb=[q commandBuffer];id<MTLComputeCommandEncoder> e=[cb computeCommandEncoderWithDescriptor:pd];[e setComputePipelineState:p];
    if(kind==0){for(int i=0;i<9;i++)[e setBuffer:b[i] offset:0 atIndex:i];}
    else if(kind==1){[e setBuffer:b[0] offset:0 atIndex:0];[e setBuffer:b[3] offset:0 atIndex:1];[e setBytes:&residual_norm length:sizeof(residual_norm) atIndex:2];[e setBuffer:b[9] offset:0 atIndex:3];}
    else {[e setBuffer:b[3] offset:0 atIndex:0];[e setBuffer:b[10] offset:0 atIndex:1];[e setBuffer:b[7] offset:0 atIndex:2];[e setBuffer:b[6] offset:0 atIndex:3];[e setBuffer:b[8] offset:0 atIndex:4];}
    [e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[e endEncoding];[cb commit];[cb waitUntilCompleted];NSData *data=[sb resolveCounterRange:NSMakeRange(0,2)];if(data.length!=2*sizeof(MTLCounterResultTimestamp)){fprintf(stderr,"invalid resolve\n");exit(1);}const MTLCounterResultTimestamp *r=data.bytes;if(!r[0].timestamp||r[1].timestamp<=r[0].timestamp){fprintf(stderr,"invalid timestamp pair\n");exit(1);}return r[1].timestamp-r[0].timestamp;
}

int main(void){@autoreleasepool{
    id<MTLDevice> d=MTLCreateSystemDefaultDevice();if(!d||![d supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]){puts("UNSUPPORTED");return 2;}id<MTLCounterSet> cs=nil;for(id<MTLCounterSet> s in d.counterSets)if([s.name caseInsensitiveCompare:@"timestamp"]==NSOrderedSame)cs=s;if(!cs){puts("UNSUPPORTED timestamp set");return 2;}
    NSError *err=nil;NSString *base=[NSString stringWithContentsOfFile:@"src/models/moe_kernel.metal" encoding:NSUTF8StringEncoding error:&err];NSString *src=[base stringByAppendingString:[NSString stringWithUTF8String:variants]];MTLCompileOptions *o=[MTLCompileOptions new];o.languageVersion=MTLLanguageVersion3_1;id<MTLLibrary> l=[d newLibraryWithSource:src options:o error:&err];if(!l){fprintf(stderr,"compile failed: %s\n",err.localizedDescription.UTF8String);return 1;}
    id<MTLComputePipelineState> p[3]={pipeline(d,l,@"mhc_pre_split_weighted_sum_norm"),pipeline(d,l,@"mhc_pre_diag_dot_only"),pipeline(d,l,@"mhc_pre_diag_phase_b_only")};id<MTLCommandQueue> q=[d newCommandQueue];
    size_t sizes[11]={MIX3*MHC_H*4,MIX3*4,3*4,MHC_H*4,HC*4,HC*HC*4,DIM*4,DIM*4,DIM*2,MIX3*4,HC*4};id<MTLBuffer> __strong b[11];for(int i=0;i<11;i++){b[i]=[d newBufferWithLength:sizes[i] options:MTLResourceStorageModeShared];memset(b[i].contents,0,sizes[i]);}
    float *fw=b[0].contents,*basev=b[1].contents,*scale=b[2].contents,*res=b[3].contents,*nw=b[7].contents,*pre=b[10].contents;for(size_t i=0;i<MIX3*MHC_H;i++)fw[i]=nextf();for(size_t i=0;i<MHC_H;i++)res[i]=nextf();for(uint i=0;i<MIX3;i++)basev[i]=nextf();scale[0]=scale[1]=scale[2]=0.1f;for(uint i=0;i<DIM;i++)nw[i]=1.0f+nextf();for(uint i=0;i<HC;i++)pre[i]=0.5f+nextf();
    double ss=0.0;for(size_t i=0;i<MHC_H;i++)ss+=(double)res[i]*res[i];float residual_norm=1.0f/sqrtf((float)(ss/(double)MHC_H)+1e-6f);
    for(int i=0;i<WARMUP;i++)for(int step=0;step<3;step++){int k=(i+step)%3;(void)run_one(d,q,p[k],cs,k,residual_norm,b);}
    uint16_t first[DIM];bool saved=false;uint64_t t[3][REPS];
    for(int i=0;i<REPS;i++)for(int step=0;step<3;step++){int k=(i+step)%3;t[k][i]=run_one(d,q,p[k],cs,k,residual_norm,b);if(k==0){uint16_t *out=b[8].contents;for(uint j=0;j<DIM;j++)if((out[j]&0x7f80u)==0x7f80u){puts("nonfinite output");return 1;}if(!saved){memcpy(first,out,sizeof(first));saved=true;}else if(memcmp(first,out,sizeof(first))!=0){puts("non-repeatable output");return 1;}}}
    if(!check_reference(b,residual_norm)){puts("production fused/dot CPU oracle mismatch");return 1;}
    const char *names[3]={"full_raw_ns","dot_raw_ns","phase_b_raw_ns"};for(int k=0;k<3;k++){printf("%s=[",names[k]);for(int i=0;i<REPS;i++)printf("%s%llu",i?",":"",(unsigned long long)t[k][i]);puts("]");qsort(t[k],REPS,sizeof(uint64_t),cmp_u64);}
    double ratio=(double)t[1][REPS/2]/(double)t[0][REPS/2];printf("full_median_ns=%llu dot_median_ns=%llu phase_b_median_ns=%llu dot_full_ratio=%.6f reps=%d topology=1cb/1encoder/1dispatch/1commit/1wait order=rotating_interleaved\n",(unsigned long long)t[0][REPS/2],(unsigned long long)t[1][REPS/2],(unsigned long long)t[2][REPS/2],ratio,REPS);printf("process_id=%d oracle_status=PASS\n",getpid());printf("RESULT: %s (isolated_ratio_threshold=0.70; non_additive)\n",ratio>=0.70?"OPERATIONAL_CANDIDATE_RETAINED":"OPERATIONAL_CANDIDATE_REJECTED");return ratio>=0.70?0:1;
}}
