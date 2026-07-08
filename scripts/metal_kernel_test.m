// Standalone Metal kernel test harness for the S2-S4 attention kernels.
// Runtime-compiles src/models/moe_kernel.metal and exercises each new kernel
// with small inputs, comparing against a CPU reference computed here.
//
// Build & run:
//   clang -framework Metal -framework Foundation -fobjc-arc \
//     scripts/metal_kernel_test.m -o /tmp/mkt && /tmp/mkt
//
// This gives a ~2s feedback loop for Metal syntax/binding/index bugs, instead
// of the ~50s server load required to exercise kernels through engine.c.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <math.h>
#import <stdio.h>
#import <stdlib.h>

static id<MTLDevice> dev;
static id<MTLCommandQueue> queue;
static id<MTLLibrary> lib;

static id<MTLComputePipelineState> mkpipe(const char *name) {
    NSError *err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
    if (!fn) { printf("FAIL: no function %s\n", name); exit(1); }
    id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!p) { printf("FAIL: pipeline %s: %s\n", name, [[err localizedDescription] UTF8String]); exit(1); }
    return p;
}

static id<MTLBuffer> buf(void *data, size_t len) {
    return [dev newBufferWithBytes:data length:len options:MTLResourceStorageModeShared];
}
static id<MTLBuffer> bufz(size_t len) {
    return [dev newBufferWithLength:len options:MTLResourceStorageModeShared];
}

static int g_fail = 0;
static void check(const char *what, float maxd, float tol) {
    printf("  %-28s max_abs_diff=%.3e  %s\n", what, maxd, maxd < tol ? "OK" : "FAIL");
    if (maxd >= tol) g_fail = 1;
}

// ---- test: rms_norm_rows (weightless + weighted) ----
static void test_rms_norm_rows(void) {
    const uint n_rows = 3, row_dim = 512;
    float *x = malloc(sizeof(float) * n_rows * row_dim);
    float *w = malloc(sizeof(float) * row_dim);
    for (uint i = 0; i < n_rows * row_dim; i++) x[i] = ((float)(i % 17) - 8.0f) * 0.1f;
    for (uint i = 0; i < row_dim; i++) w[i] = 1.0f + 0.01f * (float)(i % 5);
    float eps = 1e-6f;

    id<MTLComputePipelineState> p = mkpipe("rms_norm_rows");
    id<MTLBuffer> bx = buf(x, sizeof(float) * n_rows * row_dim);
    id<MTLBuffer> bw = buf(w, sizeof(float) * row_dim);
    id<MTLBuffer> bo = bufz(sizeof(float) * n_rows * row_dim);

    for (int weighted = 0; weighted <= 1; weighted++) {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:p];
        [e setBuffer:bx offset:0 atIndex:0];
        [e setBuffer:bw offset:0 atIndex:1];
        [e setBuffer:bo offset:0 atIndex:2];
        uint rd = row_dim; uint hw = weighted;
        [e setBytes:&rd length:4 atIndex:3];
        [e setBytes:&eps length:4 atIndex:4];
        [e setBytes:&hw length:4 atIndex:5];
        [e dispatchThreadgroups:MTLSizeMake(n_rows,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [e endEncoding];
        [cb commit]; [cb waitUntilCompleted];

        float *out = [bo contents];
        float maxd = 0;
        for (uint r = 0; r < n_rows; r++) {
            double ss = 0; for (uint i = 0; i < row_dim; i++) { float v = x[r*row_dim+i]; ss += (double)v*v; }
            float rms = 1.0f / sqrtf((float)(ss/row_dim) + eps);
            for (uint i = 0; i < row_dim; i++) {
                float ref = x[r*row_dim+i] * rms * (weighted ? w[i] : 1.0f);
                float d = fabsf(ref - out[r*row_dim+i]);
                if (d > maxd) maxd = d;
            }
        }
        check(weighted ? "rms_norm_rows (weighted)" : "rms_norm_rows (weightless)", maxd, 1e-4f);
    }
    free(x); free(w);
}

// ---- test: rope_tail_interleaved ----
static void test_rope(void) {
    const uint n_heads = 2, head_dim = 16, rope_dim = 8, nope = head_dim - rope_dim;
    uint half = rope_dim / 2;
    float *q = malloc(sizeof(float) * n_heads * head_dim);
    for (uint i = 0; i < n_heads * head_dim; i++) q[i] = 0.1f * (float)((i % 11) - 5);
    float cosv[4], sinv[4];
    for (uint i = 0; i < half; i++) { cosv[i] = cosf(0.3f * (i+1)); sinv[i] = sinf(0.3f * (i+1)); }

    float *ref = malloc(sizeof(float) * n_heads * head_dim);
    memcpy(ref, q, sizeof(float) * n_heads * head_dim);
    for (uint h = 0; h < n_heads; h++)
        for (uint i = 0; i < half; i++) {
            uint j0 = nope + 2*i, j1 = nope + 2*i + 1;
            float x0 = q[h*head_dim+j0], x1 = q[h*head_dim+j1];
            ref[h*head_dim+j0] = x0*cosv[i] - x1*sinv[i];
            ref[h*head_dim+j1] = x0*sinv[i] + x1*cosv[i];
        }

    id<MTLComputePipelineState> p = mkpipe("rope_tail_interleaved");
    id<MTLBuffer> bq = buf(q, sizeof(float) * n_heads * head_dim);
    id<MTLBuffer> bc = buf(cosv, sizeof(float)*half);
    id<MTLBuffer> bs = buf(sinv, sizeof(float)*half);
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:p];
    [e setBuffer:bq offset:0 atIndex:0];
    [e setBuffer:bc offset:0 atIndex:1];
    [e setBuffer:bs offset:0 atIndex:2];
    uint nh=n_heads, hd=head_dim, nd=nope, rd=rope_dim, inv=0;
    [e setBytes:&nh length:4 atIndex:3];
    [e setBytes:&hd length:4 atIndex:4];
    [e setBytes:&nd length:4 atIndex:5];
    [e setBytes:&rd length:4 atIndex:6];
    [e setBytes:&inv length:4 atIndex:7];
    [e dispatchThreads:MTLSizeMake(n_heads*half,1,1) threadsPerThreadgroup:MTLSizeMake(8,1,1)];
    [e endEncoding];
    [cb commit]; [cb waitUntilCompleted];

    float *out = [bq contents]; float maxd = 0;
    for (uint i = 0; i < n_heads*head_dim; i++) { float d = fabsf(ref[i]-out[i]); if (d>maxd) maxd=d; }
    check("rope_tail_interleaved", maxd, 1e-5f);
    free(q); free(ref);
}

// ---- test: mla_sdpa_decode (online softmax + sink, MQA) ----
static void test_sdpa(void) {
    const uint n_heads = 4, head_dim = 64, n_kv = 5;
    float scale = 1.0f / sqrtf((float)head_dim);
    float *q = malloc(sizeof(float)*n_heads*head_dim);
    float *kv = malloc(sizeof(float)*n_kv*head_dim);
    float *sinks = malloc(sizeof(float)*n_heads);
    for (uint i=0;i<n_heads*head_dim;i++) q[i]=0.1f*(float)((i%13)-6);
    for (uint i=0;i<n_kv*head_dim;i++) kv[i]=0.1f*(float)((i%7)-3);
    for (uint h=0;h<n_heads;h++) sinks[h]=0.2f*(float)h - 0.3f;

    // CPU reference: full softmax with sink folded into denominator
    float *ref = malloc(sizeof(float)*n_heads*head_dim);
    for (uint h=0;h<n_heads;h++) {
        float m=-INFINITY;
        float *sc = malloc(sizeof(float)*n_kv);
        for (uint k=0;k<n_kv;k++){ double dot=0; for(uint d=0;d<head_dim;d++) dot+=(double)q[h*head_dim+d]*kv[k*head_dim+d]; sc[k]=(float)dot*scale; if(sc[k]>m)m=sc[k]; }
        if (sinks[h]>m) m=sinks[h];
        double s=0; for(uint k=0;k<n_kv;k++) s+=exp(sc[k]-m); s+=exp(sinks[h]-m);
        for(uint d=0;d<head_dim;d++){ double acc=0; for(uint k=0;k<n_kv;k++) acc+=exp(sc[k]-m)*kv[k*head_dim+d]; ref[h*head_dim+d]=(float)(acc/s); }
        free(sc);
    }

    id<MTLComputePipelineState> p = mkpipe("mla_sdpa_decode");
    id<MTLBuffer> bq=buf(q,sizeof(float)*n_heads*head_dim);
    id<MTLBuffer> bkv=buf(kv,sizeof(float)*n_kv*head_dim);
    id<MTLBuffer> bsk=buf(sinks,sizeof(float)*n_heads);
    id<MTLBuffer> bo=bufz(sizeof(float)*n_heads*head_dim);
    id<MTLCommandBuffer> cb=[queue commandBuffer];
    id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
    [e setComputePipelineState:p];
    [e setBuffer:bq offset:0 atIndex:0];
    [e setBuffer:bkv offset:0 atIndex:1];
    [e setBuffer:bsk offset:0 atIndex:2];
    [e setBuffer:bo offset:0 atIndex:3];
    uint nh=n_heads,hd=head_dim,nk=n_kv;
    [e setBytes:&nh length:4 atIndex:4];
    [e setBytes:&hd length:4 atIndex:5];
    [e setBytes:&nk length:4 atIndex:6];
    [e setBytes:&scale length:4 atIndex:7];
    [e dispatchThreadgroups:MTLSizeMake(n_heads,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    [e endEncoding];
    [cb commit]; [cb waitUntilCompleted];

    float *out=[bo contents]; float maxd=0;
    for (uint i=0;i<n_heads*head_dim;i++){ float d=fabsf(ref[i]-out[i]); if(d>maxd)maxd=d; }
    check("mla_sdpa_decode (+sink)", maxd, 2e-4f);
    free(q); free(kv); free(sinks); free(ref);
}

// ---- test: mla_sdpa_decode_f16 (f16 KV cache, f32 Q, f32 output) ----
static void test_sdpa_f16(void) {
    const uint n_heads = 4, head_dim = 64, n_kv = 5;
    float scale = 1.0f / sqrtf((float)head_dim);
    float *q = malloc(sizeof(float)*n_heads*head_dim);
    float *kv_f = malloc(sizeof(float)*n_kv*head_dim);
    uint16_t *kv = malloc(sizeof(uint16_t)*n_kv*head_dim);
    float *sinks = malloc(sizeof(float)*n_heads);
    for (uint i=0;i<n_heads*head_dim;i++) q[i]=0.1f*(float)((i%13)-6);
    for (uint i=0;i<n_kv*head_dim;i++) kv_f[i]=0.1f*(float)((i%7)-3);
    for (uint i=0;i<n_kv*head_dim;i++) {
        _Float16 h = (_Float16)kv_f[i];
        kv[i] = *(uint16_t *)&h;
    }
    for (uint h=0;h<n_heads;h++) sinks[h]=0.2f*(float)h - 0.3f;

    // CPU reference: full softmax with sink folded into denominator
    float *ref = malloc(sizeof(float)*n_heads*head_dim);
    for (uint h=0;h<n_heads;h++) {
        float m=-INFINITY;
        float *sc = malloc(sizeof(float)*n_kv);
        for (uint k=0;k<n_kv;k++){ double dot=0; for(uint d=0;d<head_dim;d++) dot+=(double)q[h*head_dim+d]*kv_f[k*head_dim+d]; sc[k]=(float)dot*scale; if(sc[k]>m)m=sc[k]; }
        if (sinks[h]>m) m=sinks[h];
        double s=0; for(uint k=0;k<n_kv;k++) s+=exp(sc[k]-m); s+=exp(sinks[h]-m);
        for(uint d=0;d<head_dim;d++){ double acc=0; for(uint k=0;k<n_kv;k++) acc+=exp(sc[k]-m)*kv_f[k*head_dim+d]; ref[h*head_dim+d]=(float)(acc/s); }
        free(sc);
    }

    id<MTLComputePipelineState> p = mkpipe("mla_sdpa_decode_f16");
    id<MTLBuffer> bq=buf(q,sizeof(float)*n_heads*head_dim);
    id<MTLBuffer> bkv=buf(kv,sizeof(uint16_t)*n_kv*head_dim);
    id<MTLBuffer> bsk=buf(sinks,sizeof(float)*n_heads);
    id<MTLBuffer> bo=bufz(sizeof(float)*n_heads*head_dim);
    id<MTLCommandBuffer> cb=[queue commandBuffer];
    id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
    [e setComputePipelineState:p];
    [e setBuffer:bq offset:0 atIndex:0];
    [e setBuffer:bkv offset:0 atIndex:1];
    [e setBuffer:bsk offset:0 atIndex:2];
    [e setBuffer:bo offset:0 atIndex:3];
    uint nh=n_heads,hd=head_dim,nk=n_kv;
    [e setBytes:&nh length:4 atIndex:4];
    [e setBytes:&hd length:4 atIndex:5];
    [e setBytes:&nk length:4 atIndex:6];
    [e setBytes:&scale length:4 atIndex:7];
    [e dispatchThreadgroups:MTLSizeMake(n_heads,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    [e endEncoding];
    [cb commit]; [cb waitUntilCompleted];

    float *out=[bo contents]; float maxd=0;
    for (uint i=0;i<n_heads*head_dim;i++){ float d=fabsf(ref[i]-out[i]); if(d>maxd)maxd=d; }
    check("mla_sdpa_decode_f16 (+sink)", maxd, 2e-4f);
    free(q); free(kv_f); free(kv); free(sinks); free(ref);
}

// ---- test: mla_sdpa_decode_f16in_f16out with multi-kv (n_kv=5) ----
static void test_sdpa_f16in_f16out_multi_kv(void) {
    const uint n_heads = 4, head_dim = 64, n_kv = 5;
    float scale = 1.0f / sqrtf((float)head_dim);
    float *q = malloc(sizeof(float)*n_heads*head_dim);
    float *kv_f = malloc(sizeof(float)*n_kv*head_dim);
    float *sinks = malloc(sizeof(float)*n_heads);
    for (uint i=0;i<n_heads*head_dim;i++) q[i]=0.1f*(float)((i%13)-6);
    for (uint i=0;i<n_kv*head_dim;i++) kv_f[i]=0.1f*(float)((i%7)-3);
    for (uint h=0;h<n_heads;h++) sinks[h]=0.2f*(float)h - 0.3f;

    uint16_t *q_h = malloc(sizeof(uint16_t)*n_heads*head_dim);
    uint16_t *kv_h = malloc(sizeof(uint16_t)*n_kv*head_dim);
    for (uint i=0;i<n_heads*head_dim;i++) { _Float16 h = (_Float16)q[i]; q_h[i] = *(uint16_t *)&h; }
    for (uint i=0;i<n_kv*head_dim;i++) { _Float16 h = (_Float16)kv_f[i]; kv_h[i] = *(uint16_t *)&h; }

    // CPU reference: online softmax with sink (using last KV as sink KV)
    float *ref = malloc(sizeof(float)*n_heads*head_dim);
    for (uint h=0;h<n_heads;h++) {
        float m=-INFINITY; float s=0.0f;
        float acc[64]; memset(acc,0,sizeof(float)*head_dim);
        for (uint k=0;k<n_kv;k++) {
            double dot=0; for(uint d=0;d<head_dim;d++) dot+=(double)q[h*head_dim+d]*kv_f[k*head_dim+d];
            float score=(float)dot*scale;
            float m_new=fmaxf(m,score);
            float corr=(m==-INFINITY)?0.0f:expf(m-m_new);
            float p=expf(score-m_new);
            for(uint d=0;d<head_dim;d++) acc[d]=acc[d]*corr+p*kv_f[k*head_dim+d];
            s=s*corr+p; m=m_new;
        }
        // Sink
        float sink=sinks[h];
        float m_new=fmaxf(m,sink);
        float corr=(m==-INFINITY)?0.0f:expf(m-m_new);
        float p=expf(sink-m_new);
        for(uint d=0;d<head_dim;d++) acc[d]=acc[d]*corr+p*kv_f[(n_kv-1)*head_dim+d];
        s=s*corr+p;
        for(uint d=0;d<head_dim;d++) ref[h*head_dim+d]=acc[d]/s;
    }

    id<MTLComputePipelineState> p = mkpipe("mla_sdpa_decode_f16in_f16out");
    id<MTLBuffer> bq=buf(q_h,sizeof(uint16_t)*n_heads*head_dim);
    id<MTLBuffer> bkv=buf(kv_h,sizeof(uint16_t)*n_kv*head_dim);
    id<MTLBuffer> bsk=buf(sinks,sizeof(float)*n_heads);
    id<MTLBuffer> bo=bufz(sizeof(uint16_t)*n_heads*head_dim);
    id<MTLCommandBuffer> cb=[queue commandBuffer];
    id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
    [e setComputePipelineState:p];
    [e setBuffer:bq offset:0 atIndex:0];
    [e setBuffer:bkv offset:0 atIndex:1];
    [e setBuffer:bsk offset:0 atIndex:2];
    [e setBuffer:bo offset:0 atIndex:3];
    uint nh=n_heads,hd=head_dim,nk=n_kv;
    [e setBytes:&nh length:4 atIndex:4];
    [e setBytes:&hd length:4 atIndex:5];
    [e setBytes:&nk length:4 atIndex:6];
    [e setBytes:&scale length:4 atIndex:7];
    [e dispatchThreadgroups:MTLSizeMake(n_heads,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    [e endEncoding];
    [cb commit]; [cb waitUntilCompleted];

    uint16_t *out_h=(uint16_t *)[bo contents]; float maxd=0;
    for (uint i=0;i<n_heads*head_dim;i++){
        _Float16 h = *(_Float16 *)&out_h[i];
        float d=fabsf(ref[i]-(float)h); if(d>maxd)maxd=d;
    }
    check("mla_sdpa_decode_f16in_f16out (multi-kv)", maxd, 2e-4f);
    free(q); free(kv_f); free(sinks); free(q_h); free(kv_h); free(ref);
}

// ---- test: dequant_matvec_affine (w = scale*nibble + bias) ----
static void test_dequant_affine(void) {
    const uint out_dim = 5, in_dim = 128, gs = 64;
    uint ng = in_dim / gs, pcols = in_dim / 8;
    uint32_t *packed = malloc(sizeof(uint32_t) * out_dim * pcols);
    float *scales = malloc(sizeof(float) * out_dim * ng);
    float *biases = malloc(sizeof(float) * out_dim * ng);
    float *x = malloc(sizeof(float) * in_dim);
    for (uint i = 0; i < out_dim * pcols; i++) packed[i] = (uint32_t)(i * 2654435761u);
    for (uint i = 0; i < out_dim * ng; i++) { scales[i] = 0.01f + 0.001f*(i%7); biases[i] = -0.05f + 0.002f*(i%5); }
    for (uint i = 0; i < in_dim; i++) x[i] = 0.1f * (float)((i%9)-4);

    // CPU reference mirroring the kernel
    float *ref = malloc(sizeof(float)*out_dim);
    for (uint r = 0; r < out_dim; r++) {
        float acc = 0;
        for (uint g = 0; g < ng; g++) {
            float sc = scales[r*ng+g], bi = biases[r*ng+g];
            for (uint p = 0; p < gs/8; p++) {
                uint32_t pw = packed[r*pcols + g*(gs/8) + p];
                uint xb = g*gs + p*8;
                for (uint i = 0; i < 8; i++) {
                    float nib = (float)((pw >> (i*4)) & 0xF);
                    acc += (sc*nib + bi) * x[xb+i];
                }
            }
        }
        ref[r] = acc;
    }

    id<MTLComputePipelineState> p = mkpipe("dequant_matvec_affine");
    id<MTLBuffer> bw=buf(packed,sizeof(uint32_t)*out_dim*pcols);
    id<MTLBuffer> bs=buf(scales,sizeof(float)*out_dim*ng);
    id<MTLBuffer> bb=buf(biases,sizeof(float)*out_dim*ng);
    id<MTLBuffer> bx=buf(x,sizeof(float)*in_dim);
    id<MTLBuffer> bo=bufz(sizeof(float)*out_dim);
    id<MTLCommandBuffer> cb=[queue commandBuffer];
    id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
    [e setComputePipelineState:p];
    [e setBuffer:bw offset:0 atIndex:0];
    [e setBuffer:bs offset:0 atIndex:1];
    [e setBuffer:bb offset:0 atIndex:2];
    [e setBuffer:bx offset:0 atIndex:3];
    [e setBuffer:bo offset:0 atIndex:4];
    uint od=out_dim, id_=in_dim, g=gs;
    [e setBytes:&od length:4 atIndex:5];
    [e setBytes:&id_ length:4 atIndex:6];
    [e setBytes:&g length:4 atIndex:7];
    [e dispatchThreads:MTLSizeMake(out_dim,1,1) threadsPerThreadgroup:MTLSizeMake(out_dim,1,1)];
    [e endEncoding];
    [cb commit]; [cb waitUntilCompleted];

    float *out=[bo contents]; float maxd=0;
    for (uint i=0;i<out_dim;i++){ float d=fabsf(ref[i]-out[i]); if(d>maxd)maxd=d; }
    check("dequant_matvec_affine", maxd, 1e-4f);
    free(packed); free(scales); free(biases); free(x); free(ref);
}

// ---- test: dequant_matvec_int8_e8m0 ----
static void test_dequant_int8_e8m0(void) {
    printf("test_dequant_int8_e8m0:\n");
    // Small test: out_dim=16, in_dim=64, block_size=16 → num_groups=4
    const uint out_dim = 16, in_dim = 64, block_size_val = 16;
    const uint num_groups = in_dim / block_size_val;

    int8_t *W = malloc(out_dim * in_dim);
    uint8_t *scales = malloc(out_dim * num_groups);
    float *x = malloc(sizeof(float) * in_dim);
    for (uint i = 0; i < out_dim * in_dim; i++) W[i] = (int8_t)((i * 7 + 3) % 251 - 125);
    for (uint i = 0; i < out_dim * num_groups; i++) scales[i] = 120 + (i % 8); // E8M0: 2^(120..127 - 127)
    for (uint i = 0; i < in_dim; i++) x[i] = ((float)(i % 11) - 5.0f) * 0.1f;

    // CPU reference
    float ref[16];
    for (uint r = 0; r < out_dim; r++) {
        double acc = 0;
        for (uint g = 0; g < num_groups; g++) {
            // E8M0 decode: 2^(byte - 127)
            uint8_t sb = scales[r * num_groups + g];
            uint32_t fbits = (uint32_t)sb << 23;
            float sf = *(float*)&fbits;
            for (uint i = 0; i < block_size_val; i++) {
                uint idx = g * block_size_val + i;
                acc += (double)W[r * in_dim + idx] * sf * (double)x[idx];
            }
        }
        ref[r] = (float)acc;
    }

    // GPU
    id<MTLComputePipelineState> p = mkpipe("dequant_matvec_int8_e8m0");
    id<MTLBuffer> bW = buf(W, out_dim * in_dim);
    id<MTLBuffer> bS = buf(scales, out_dim * num_groups);
    id<MTLBuffer> bX = buf(x, sizeof(float) * in_dim);
    id<MTLBuffer> bO = bufz(sizeof(float) * out_dim);
    uint od = out_dim, id_val = in_dim, bs = block_size_val;

    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:p];
    [e setBuffer:bW offset:0 atIndex:0];
    [e setBuffer:bS offset:0 atIndex:1];
    [e setBuffer:bX offset:0 atIndex:2];
    [e setBuffer:bO offset:0 atIndex:3];
    [e setBytes:&od length:4 atIndex:4];
    [e setBytes:&id_val length:4 atIndex:5];
    [e setBytes:&bs length:4 atIndex:6];
    [e dispatchThreadgroups:MTLSizeMake((out_dim + 7) / 8, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    [cb commit]; [cb waitUntilCompleted];

    float *out_data = [bO contents];
    float maxd = 0;
    for (uint i = 0; i < out_dim; i++) {
        float d = fabsf(ref[i] - out_data[i]);
        if (d > maxd) maxd = d;
    }
    check("int8_e8m0 matvec", maxd, 1e-4f);
    free(W); free(scales); free(x);
}

// ---- test: fused_gate_up_swiglu_int8_e8m0 ----
static void test_fused_gate_up_swiglu_int8_e8m0(void) {
    printf("test_fused_gate_up_swiglu_int8_e8m0:\n");
    const uint out_dim = 8, in_dim = 64, block_size_val = 16;
    const uint num_groups = in_dim / block_size_val;

    int8_t *gate_W = malloc(out_dim * in_dim);
    uint8_t *gate_s = malloc(out_dim * num_groups);
    int8_t *up_W = malloc(out_dim * in_dim);
    uint8_t *up_s = malloc(out_dim * num_groups);
    float *x = malloc(sizeof(float) * in_dim);

    for (uint i = 0; i < out_dim * in_dim; i++) { gate_W[i] = (int8_t)((i * 3 + 1) % 127 - 63); up_W[i] = (int8_t)((i * 5 + 7) % 127 - 63); }
    for (uint i = 0; i < out_dim * num_groups; i++) { gate_s[i] = 121 + (i % 4); up_s[i] = 120 + (i % 5); }
    for (uint i = 0; i < in_dim; i++) x[i] = ((float)(i % 9) - 4.0f) * 0.05f;

    // CPU reference
    float ref[8];
    for (uint r = 0; r < out_dim; r++) {
        double gate_val = 0, up_val = 0;
        for (uint g = 0; g < num_groups; g++) {
            uint32_t gb = (uint32_t)gate_s[r * num_groups + g] << 23;
            uint32_t ub = (uint32_t)up_s[r * num_groups + g] << 23;
            float gsf = *(float*)&gb, usf = *(float*)&ub;
            for (uint i = 0; i < block_size_val; i++) {
                uint idx = g * block_size_val + i;
                gate_val += (double)gate_W[r * in_dim + idx] * gsf * (double)x[idx];
                up_val += (double)up_W[r * in_dim + idx] * usf * (double)x[idx];
            }
        }
        float g_c = fminf((float)gate_val, 10.0f);
        float u_c = fminf(fmaxf((float)up_val, -10.0f), 10.0f);
        float act = g_c / (1.0f + expf(-g_c));
        ref[r] = act * u_c;
    }

    // GPU
    id<MTLComputePipelineState> p = mkpipe("fused_gate_up_swiglu_int8_e8m0");
    id<MTLBuffer> bGW = buf(gate_W, out_dim * in_dim);
    id<MTLBuffer> bGS = buf(gate_s, out_dim * num_groups);
    id<MTLBuffer> bUW = buf(up_W, out_dim * in_dim);
    id<MTLBuffer> bUS = buf(up_s, out_dim * num_groups);
    id<MTLBuffer> bX = buf(x, sizeof(float) * in_dim);
    id<MTLBuffer> bO = bufz(sizeof(float) * out_dim);
    uint od = out_dim, id_val = in_dim, bs = block_size_val;

    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:p];
    [enc setBuffer:bGW offset:0 atIndex:0];
    [enc setBuffer:bGS offset:0 atIndex:1];
    [enc setBuffer:bUW offset:0 atIndex:2];
    [enc setBuffer:bUS offset:0 atIndex:3];
    [enc setBuffer:bX offset:0 atIndex:4];
    [enc setBuffer:bO offset:0 atIndex:5];
    [enc setBytes:&od length:4 atIndex:6];
    [enc setBytes:&id_val length:4 atIndex:7];
    [enc setBytes:&bs length:4 atIndex:8];
    [enc dispatchThreadgroups:MTLSizeMake((out_dim + 7) / 8, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    [cb commit]; [cb waitUntilCompleted];

    float *out_data = [bO contents];
    float maxd = 0;
    for (uint i = 0; i < out_dim; i++) {
        float d = fabsf(ref[i] - out_data[i]);
        if (d > maxd) maxd = d;
    }
    check("int8_e8m0 fused_gate_up_swiglu", maxd, 1e-4f);
    free(gate_W); free(gate_s); free(up_W); free(up_s); free(x);
}

int main(int argc, char **argv) {
    const char *src_path = (argc > 1) ? argv[1] : "src/models/moe_kernel.metal";
    dev = MTLCreateSystemDefaultDevice();
    if (!dev) { printf("no metal device\n"); return 1; }
    queue = [dev newCommandQueue];

    NSError *err = nil;
    NSString *src = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:src_path]
                                              encoding:NSUTF8StringEncoding error:&err];
    if (!src) { printf("read %s failed\n", src_path); return 1; }
    MTLCompileOptions *opts = [MTLCompileOptions new];
    opts.languageVersion = MTLLanguageVersion3_1;
    lib = [dev newLibraryWithSource:src options:opts error:&err];
    if (!lib) { printf("COMPILE FAILED:\n%s\n", [[err localizedDescription] UTF8String]); return 1; }
    printf("moe_kernel.metal compiled OK\n");

    test_rms_norm_rows();
    test_rope();
    test_sdpa();
    test_sdpa_f16();
    test_sdpa_f16in_f16out_multi_kv();
    test_dequant_affine();
    test_dequant_int8_e8m0();
    test_fused_gate_up_swiglu_int8_e8m0();

    printf(g_fail ? "\nRESULT: KERNEL TESTS FAILED\n" : "\nRESULT: ALL KERNEL TESTS PASSED\n");
    return g_fail;
}
