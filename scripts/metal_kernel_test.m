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
#import <time.h>

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

// ---- test: dequant_matvec_affine batched shared projection ----
static void test_dequant_affine(void) {
    const uint out_dim = 5, in_dim = 128, gs = 64, max_batch = 5;
    const uint ng = in_dim / gs, pcols = in_dim / 8;
    const uint batch_sizes[] = {1u, 2u, 3u, 5u};
    uint32_t *packed = malloc(sizeof(uint32_t) * out_dim * pcols);
    float *scales = malloc(sizeof(float) * out_dim * ng);
    float *biases = malloc(sizeof(float) * out_dim * ng);
    float *x = malloc(sizeof(float) * max_batch * in_dim);
    for (uint i = 0; i < out_dim * pcols; i++) {
        packed[i] = (uint32_t)(i * 2654435761u);
    }
    for (uint i = 0; i < out_dim * ng; i++) {
        scales[i] = 0.01f + 0.001f * (i % 7);
        biases[i] = -0.05f + 0.002f * (i % 5);
    }
    for (uint i = 0; i < max_batch * in_dim; i++) {
        x[i] = ((float)((int)(i % 9) - 4)) * 0.1f;
    }

    id<MTLComputePipelineState> p = mkpipe("dequant_matvec_affine");
    id<MTLBuffer> bw = buf(packed, sizeof(uint32_t) * out_dim * pcols);
    id<MTLBuffer> bs = buf(scales, sizeof(float) * out_dim * ng);
    id<MTLBuffer> bb = buf(biases, sizeof(float) * out_dim * ng);
    id<MTLBuffer> bx = buf(x, sizeof(float) * max_batch * in_dim);
    for (size_t case_index = 0u;
         case_index < sizeof(batch_sizes) / sizeof(batch_sizes[0]);
         ++case_index) {
        const uint B = batch_sizes[case_index];
        float *ref = malloc(sizeof(float) * B * out_dim);
        for (uint token = 0u; token < B; ++token) {
            for (uint r = 0; r < out_dim; r++) {
                float acc = 0.0f;
                for (uint group = 0; group < ng; group++) {
                    const float sc = scales[r * ng + group];
                    const float bi = biases[r * ng + group];
                    for (uint word = 0; word < gs / 8; word++) {
                        const uint32_t pw = packed[
                            r * pcols + group * (gs / 8) + word];
                        const uint xb = group * gs + word * 8;
                        for (uint nibble = 0; nibble < 8; nibble++) {
                            const float q = (float)(
                                (pw >> (nibble * 4)) & 0xf);
                            acc += (sc * q + bi) *
                                x[(size_t)token * in_dim + xb + nibble];
                        }
                    }
                }
                ref[(size_t)token * out_dim + r] = acc;
            }
        }
        id<MTLBuffer> bo = bufz(sizeof(float) * B * out_dim);
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:p];
        [e setBuffer:bw offset:0 atIndex:0];
        [e setBuffer:bs offset:0 atIndex:1];
        [e setBuffer:bb offset:0 atIndex:2];
        [e setBuffer:bx offset:0 atIndex:3];
        [e setBuffer:bo offset:0 atIndex:4];
        uint od = out_dim, id_ = in_dim, g = gs;
        [e setBytes:&od length:4 atIndex:5];
        [e setBytes:&id_ length:4 atIndex:6];
        [e setBytes:&g length:4 atIndex:7];
        [e dispatchThreads:MTLSizeMake(out_dim, B, 1u)
            threadsPerThreadgroup:MTLSizeMake(out_dim, 1u, 1u)];
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        float *out = [bo contents];
        float maxd = 0.0f;
        int finite = cb.status == MTLCommandBufferStatusCompleted;
        for (uint i = 0; i < B * out_dim; i++) {
            finite = finite && isfinite(out[i]);
            const float d = fabsf(ref[i] - out[i]);
            if (d > maxd) maxd = d;
        }
        char label[80];
        snprintf(label, sizeof(label), "dequant_matvec_affine B=%u", B);
        check(label, maxd, 1e-4f);
        if (!finite) g_fail = 1;
        free(ref);
    }
    free(packed);
    free(scales);
    free(biases);
    free(x);
}

static void test_shared_expert_batch_chain(void) {
    printf("test_shared_expert_batch_chain:\n");
    const uint32_t in_dim = 128u, mid_dim = 64u, out_dim = 17u;
    const uint32_t group_size = 64u, max_batch = 5u;
    const uint32_t batch_sizes[] = {1u, 2u, 3u, 5u};
    const size_t gu_words = (size_t)mid_dim * (in_dim / 8u);
    const size_t gu_parameters = (size_t)mid_dim * (in_dim / group_size);
    const size_t down_words = (size_t)out_dim * (mid_dim / 8u);
    const size_t down_parameters =
        (size_t)out_dim * (mid_dim / group_size);
    uint32_t *gate_weights = malloc(gu_words * sizeof(uint32_t));
    uint32_t *up_weights = malloc(gu_words * sizeof(uint32_t));
    uint32_t *down_weights = malloc(down_words * sizeof(uint32_t));
    float *gate_scales = malloc(gu_parameters * sizeof(float));
    float *gate_biases = malloc(gu_parameters * sizeof(float));
    float *up_scales = malloc(gu_parameters * sizeof(float));
    float *up_biases = malloc(gu_parameters * sizeof(float));
    float *down_scales = malloc(down_parameters * sizeof(float));
    float *down_biases = malloc(down_parameters * sizeof(float));
    float *input = malloc((size_t)max_batch * in_dim * sizeof(float));
    for (size_t index = 0u; index < gu_words; ++index) {
        gate_weights[index] = (uint32_t)(index * 2654435761u + 0x1234567u);
        up_weights[index] = (uint32_t)(index * 2246822519u + 0x7654321u);
    }
    for (size_t index = 0u; index < down_words; ++index) {
        down_weights[index] = (uint32_t)(index * 3266489917u + 0x2468135u);
    }
    for (size_t index = 0u; index < gu_parameters; ++index) {
        gate_scales[index] = 0.002f + (float)(index % 5u) * 0.0002f;
        gate_biases[index] = -0.01f + (float)(index % 3u) * 0.001f;
        up_scales[index] = 0.0015f + (float)(index % 7u) * 0.00015f;
        up_biases[index] = -0.008f + (float)(index % 4u) * 0.0008f;
    }
    for (size_t index = 0u; index < down_parameters; ++index) {
        down_scales[index] = 0.0025f + (float)(index % 6u) * 0.0001f;
        down_biases[index] = -0.006f + (float)(index % 5u) * 0.0006f;
    }
    for (size_t index = 0u; index < (size_t)max_batch * in_dim; ++index) {
        input[index] = (float)((int)(index % 19u) - 9) * 0.015625f;
    }

    id<MTLComputePipelineState> affine = mkpipe("dequant_matvec_affine");
    id<MTLComputePipelineState> swiglu = mkpipe("limited_swiglu");
    id<MTLBuffer> gate_weight_buffer = buf(
        gate_weights, gu_words * sizeof(uint32_t));
    id<MTLBuffer> gate_scale_buffer = buf(
        gate_scales, gu_parameters * sizeof(float));
    id<MTLBuffer> gate_bias_buffer = buf(
        gate_biases, gu_parameters * sizeof(float));
    id<MTLBuffer> up_weight_buffer = buf(
        up_weights, gu_words * sizeof(uint32_t));
    id<MTLBuffer> up_scale_buffer = buf(
        up_scales, gu_parameters * sizeof(float));
    id<MTLBuffer> up_bias_buffer = buf(
        up_biases, gu_parameters * sizeof(float));
    id<MTLBuffer> down_weight_buffer = buf(
        down_weights, down_words * sizeof(uint32_t));
    id<MTLBuffer> down_scale_buffer = buf(
        down_scales, down_parameters * sizeof(float));
    id<MTLBuffer> down_bias_buffer = buf(
        down_biases, down_parameters * sizeof(float));
    id<MTLBuffer> input_buffer = buf(
        input, (size_t)max_batch * in_dim * sizeof(float));

    for (size_t case_index = 0u;
         case_index < sizeof(batch_sizes) / sizeof(batch_sizes[0]);
         ++case_index) {
        const uint32_t B = batch_sizes[case_index];
        const size_t output_values = (size_t)B * out_dim;
        float *reference = malloc(output_values * sizeof(float));
        float *gate_reference = malloc((size_t)B * mid_dim * sizeof(float));
        for (uint32_t token = 0u; token < B; ++token) {
            for (uint32_t row = 0u; row < mid_dim; ++row) {
                float gate_sum = 0.0f, up_sum = 0.0f;
                for (uint32_t group = 0u;
                     group < in_dim / group_size;
                     ++group) {
                    for (uint32_t word = 0u; word < group_size / 8u; ++word) {
                        const uint32_t gate_word = gate_weights[
                            row * (in_dim / 8u) +
                            group * (group_size / 8u) + word];
                        const uint32_t up_word = up_weights[
                            row * (in_dim / 8u) +
                            group * (group_size / 8u) + word];
                        const uint32_t base = group * group_size + word * 8u;
                        for (uint32_t nibble = 0u; nibble < 8u; ++nibble) {
                            const float value = input[
                                (size_t)token * in_dim + base + nibble];
                            gate_sum += (gate_scales[
                                row * (in_dim / group_size) + group] *
                                (float)((gate_word >> (nibble * 4u)) & 0xfu) +
                                gate_biases[
                                row * (in_dim / group_size) + group]) * value;
                            up_sum += (up_scales[
                                row * (in_dim / group_size) + group] *
                                (float)((up_word >> (nibble * 4u)) & 0xfu) +
                                up_biases[
                                row * (in_dim / group_size) + group]) * value;
                        }
                    }
                }
                const float gate_clamped = fminf(gate_sum, 10.0f);
                const float up_clamped = fminf(fmaxf(up_sum, -10.0f), 10.0f);
                gate_reference[(size_t)token * mid_dim + row] =
                    (gate_clamped / (1.0f + expf(-gate_clamped))) * up_clamped;
            }
            for (uint32_t row = 0u; row < out_dim; ++row) {
                float sum = 0.0f;
                for (uint32_t word = 0u; word < mid_dim / 8u; ++word) {
                    const uint32_t packed = down_weights[
                        row * (mid_dim / 8u) + word];
                    for (uint32_t nibble = 0u; nibble < 8u; ++nibble) {
                        sum += (down_scales[row] *
                            (float)((packed >> (nibble * 4u)) & 0xfu) +
                            down_biases[row]) * gate_reference[
                            (size_t)token * mid_dim + word * 8u + nibble];
                    }
                }
                reference[(size_t)token * out_dim + row] = sum;
            }
        }

        id<MTLBuffer> gate_buffer = bufz(
            (size_t)B * mid_dim * sizeof(float));
        id<MTLBuffer> up_buffer = bufz(
            (size_t)B * mid_dim * sizeof(float));
        id<MTLBuffer> output_buffer = bufz(output_values * sizeof(float));
        id<MTLCommandBuffer> command = [queue commandBuffer];
        const id<MTLBuffer> projection_weights[3] = {
            gate_weight_buffer, up_weight_buffer, down_weight_buffer};
        const id<MTLBuffer> projection_scales[3] = {
            gate_scale_buffer, up_scale_buffer, down_scale_buffer};
        const id<MTLBuffer> projection_biases[3] = {
            gate_bias_buffer, up_bias_buffer, down_bias_buffer};
        const id<MTLBuffer> projection_inputs[3] = {
            input_buffer, input_buffer, gate_buffer};
        const id<MTLBuffer> projection_outputs[3] = {
            gate_buffer, up_buffer, output_buffer};
        const uint32_t projection_out[3] = {mid_dim, mid_dim, out_dim};
        const uint32_t projection_in[3] = {in_dim, in_dim, mid_dim};
        for (uint32_t projection = 0u; projection < 3u; ++projection) {
            if (projection == 2u) {
                id<MTLComputeCommandEncoder> activation =
                    [command computeCommandEncoder];
                [activation setComputePipelineState:swiglu];
                [activation setBuffer:gate_buffer offset:0 atIndex:0];
                [activation setBuffer:up_buffer offset:0 atIndex:1];
                const uint32_t count = B * mid_dim;
                [activation setBytes:&count length:sizeof(count) atIndex:2];
                [activation dispatchThreads:MTLSizeMake(count, 1u, 1u)
                    threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
                [activation endEncoding];
            }
            id<MTLComputeCommandEncoder> encoder =
                [command computeCommandEncoder];
            [encoder setComputePipelineState:affine];
            [encoder setBuffer:projection_weights[projection]
                         offset:0 atIndex:0];
            [encoder setBuffer:projection_scales[projection]
                         offset:0 atIndex:1];
            [encoder setBuffer:projection_biases[projection]
                         offset:0 atIndex:2];
            [encoder setBuffer:projection_inputs[projection]
                         offset:0 atIndex:3];
            [encoder setBuffer:projection_outputs[projection]
                         offset:0 atIndex:4];
            const uint32_t output_dim = projection_out[projection];
            const uint32_t input_dim = projection_in[projection];
            [encoder setBytes:&output_dim length:sizeof(output_dim) atIndex:5];
            [encoder setBytes:&input_dim length:sizeof(input_dim) atIndex:6];
            [encoder setBytes:&group_size length:sizeof(group_size) atIndex:7];
            [encoder dispatchThreads:MTLSizeMake(output_dim, B, 1u)
                threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
            [encoder endEncoding];
        }
        [command commit];
        [command waitUntilCompleted];
        float *actual = (float *)[output_buffer contents];
        float max_diff = 0.0f;
        int finite = command.status == MTLCommandBufferStatusCompleted;
        for (size_t index = 0u; index < output_values; ++index) {
            finite = finite && isfinite(actual[index]);
            const float diff = fabsf(actual[index] - reference[index]);
            if (diff > max_diff) max_diff = diff;
        }
        char label[80];
        snprintf(label, sizeof(label), "shared batch chain B=%u", B);
        check(label, max_diff, 1e-5f);
        if (!finite) g_fail = 1;
        free(reference);
        free(gate_reference);
    }
    free(gate_weights);
    free(up_weights);
    free(down_weights);
    free(gate_scales);
    free(gate_biases);
    free(up_scales);
    free(up_biases);
    free(down_scales);
    free(down_biases);
    free(input);
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

// ---- test: Wave-4 multi-token MXFP4 gate/up + SwiGLU ----
static const float MXFP4_LUT[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

static float e8m0_scale(uint8_t scale) {
    uint32_t bits = (uint32_t)scale << 23;
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static float round_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    const uint32_t rounding = 0x7fffu + ((bits >> 16) & 1u);
    bits = (bits + rounding) & 0xffff0000u;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static void test_bucket_mxfp4_gate_up_swiglu(void) {
    printf("test_bucket_mxfp4_gate_up_swiglu:\n");
    const uint32_t in_dim = 64u;
    const uint32_t out_dim = 17u; // exercises the 8-row output tail
    const uint32_t group_size = 32u;
    const uint32_t packed_cols = in_dim / 8u;
    const uint32_t num_groups = in_dim / group_size;
    const size_t weight_words = (size_t)out_dim * packed_cols;
    const size_t scale_count = (size_t)out_dim * num_groups;
    uint32_t *gate_weights = malloc(weight_words * sizeof(uint32_t));
    uint32_t *up_weights = malloc(weight_words * sizeof(uint32_t));
    uint8_t *gate_scales = malloc(scale_count);
    uint8_t *up_scales = malloc(scale_count);
    for (size_t word = 0u; word < weight_words; ++word) {
        uint32_t gate_word = 0u;
        uint32_t up_word = 0u;
        for (uint32_t nibble = 0u; nibble < 8u; ++nibble) {
            gate_word |= (uint32_t)((word * 5u + nibble * 3u + 1u) & 15u)
                << (nibble * 4u);
            up_word |= (uint32_t)((word * 7u + nibble * 5u + 2u) & 15u)
                << (nibble * 4u);
        }
        gate_weights[word] = gate_word;
        up_weights[word] = up_word;
    }
    for (size_t index = 0u; index < scale_count; ++index) {
        gate_scales[index] = (uint8_t)(123u + index % 5u);
        up_scales[index] = (uint8_t)(122u + index % 6u);
    }

    id<MTLComputePipelineState> pipeline =
        mkpipe("bucket_mxfp4_gate_up_swiglu");
    const uint32_t bucket_counts[] = {1u, 2u, 3u, 4u};
    for (size_t case_index = 0u;
         case_index < sizeof(bucket_counts) / sizeof(bucket_counts[0]);
         ++case_index) {
        const uint32_t M = bucket_counts[case_index];
        const size_t input_count = (size_t)M * in_dim;
        const size_t output_count = (size_t)M * out_dim;
        const size_t guard_count = 32u;
        float *input = malloc(input_count * sizeof(float));
        float *output = malloc((output_count + guard_count) * sizeof(float));
        float *reference = malloc(output_count * sizeof(float));
        for (size_t index = 0u; index < input_count; ++index) {
            input[index] = ((float)((int)(index % 19u) - 9)) * 0.03125f;
        }
        for (size_t index = 0u; index < output_count + guard_count; ++index) {
            output[index] = -23456.0f;
        }

        for (uint32_t token = 0u; token < M; ++token) {
            for (uint32_t row = 0u; row < out_dim; ++row) {
                float gate_acc = 0.0f;
                float up_acc = 0.0f;
                for (uint32_t group = 0u; group < num_groups; ++group) {
                    const float gate_sf = e8m0_scale(
                        gate_scales[(size_t)row * num_groups + group]);
                    const float up_sf = e8m0_scale(
                        up_scales[(size_t)row * num_groups + group]);
                    for (uint32_t packed_index = 0u;
                         packed_index < group_size / 8u;
                         ++packed_index) {
                        const uint32_t word_index =
                            row * packed_cols + group * (group_size / 8u) +
                            packed_index;
                        const uint32_t gate_word = gate_weights[word_index];
                        const uint32_t up_word = up_weights[word_index];
                        const uint32_t column =
                            group * group_size + packed_index * 8u;
                        for (uint32_t nibble = 0u; nibble < 8u; ++nibble) {
                            const float value =
                                input[(size_t)token * in_dim + column + nibble];
                            gate_acc += MXFP4_LUT[
                                (gate_word >> (nibble * 4u)) & 15u] *
                                gate_sf * value;
                            up_acc += MXFP4_LUT[
                                (up_word >> (nibble * 4u)) & 15u] *
                                up_sf * value;
                        }
                    }
                }
                const float gate_clamped = fminf(gate_acc, 10.0f);
                const float up_clamped = fminf(fmaxf(up_acc, -10.0f), 10.0f);
                reference[(size_t)token * out_dim + row] = round_to_bf16(
                    (gate_clamped / (1.0f + expf(-gate_clamped))) *
                    up_clamped);
            }
        }

        id<MTLBuffer> gate_weight_buffer =
            buf(gate_weights, weight_words * sizeof(uint32_t));
        id<MTLBuffer> gate_scale_buffer = buf(gate_scales, scale_count);
        id<MTLBuffer> up_weight_buffer =
            buf(up_weights, weight_words * sizeof(uint32_t));
        id<MTLBuffer> up_scale_buffer = buf(up_scales, scale_count);
        id<MTLBuffer> input_buffer = buf(input, input_count * sizeof(float));
        id<MTLBuffer> output_buffer =
            buf(output, (output_count + guard_count) * sizeof(float));
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:gate_weight_buffer offset:0 atIndex:0];
        [encoder setBuffer:gate_scale_buffer offset:0 atIndex:1];
        [encoder setBuffer:up_weight_buffer offset:0 atIndex:2];
        [encoder setBuffer:up_scale_buffer offset:0 atIndex:3];
        [encoder setBuffer:input_buffer offset:0 atIndex:4];
        [encoder setBuffer:output_buffer offset:0 atIndex:5];
        [encoder setBytes:&M length:sizeof(M) atIndex:6];
        [encoder setBytes:&out_dim length:sizeof(out_dim) atIndex:7];
        [encoder setBytes:&in_dim length:sizeof(in_dim) atIndex:8];
        [encoder setBytes:&group_size length:sizeof(group_size) atIndex:9];
        [encoder dispatchThreadgroups:
            MTLSizeMake((out_dim + 7u) / 8u, M + 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];

        float *actual = (float *)[output_buffer contents];
        float max_diff = 0.0f;
        int finite = command.status == MTLCommandBufferStatusCompleted;
        for (size_t index = 0u; index < output_count; ++index) {
            finite = finite && isfinite(actual[index]);
            const float diff = fabsf(actual[index] - reference[index]);
            if (diff > max_diff) max_diff = diff;
        }
        for (size_t index = output_count;
             index < output_count + guard_count;
             ++index) {
            finite = finite && actual[index] == -23456.0f;
        }
        char label[96];
        snprintf(label, sizeof(label),
                 "bucket_mxfp4_gate_up M=%u", M);
        check(label, max_diff, 2e-3f);
        printf("  %-28s dispatches=1 vs %u  %s\n", "finite/tail/dispatch",
               M, finite ? "OK" : "FAIL");
        if (!finite) g_fail = 1;
        free(input);
        free(output);
        free(reference);
    }
    free(gate_weights);
    free(up_weights);
    free(gate_scales);
    free(up_scales);
}

// ---- test: Wave-4 multi-token MXFP4 down projection ----
static double monotonic_ms(void) {
    struct timespec timestamp;
    clock_gettime(CLOCK_MONOTONIC, &timestamp);
    return (double)timestamp.tv_sec * 1000.0 +
        (double)timestamp.tv_nsec / 1000000.0;
}

static double median_double(double *values, size_t count) {
    for (size_t index = 1u; index < count; ++index) {
        const double value = values[index];
        size_t insertion = index;
        while (insertion > 0u && values[insertion - 1u] > value) {
            values[insertion] = values[insertion - 1u];
            --insertion;
        }
        values[insertion] = value;
    }
    return values[count / 2u];
}

static double time_mxfp4_down_dispatch(
    id<MTLComputePipelineState> bucket_pipeline,
    id<MTLComputePipelineState> scalar_pipeline,
    id<MTLBuffer> weight_buffer,
    id<MTLBuffer> scale_buffer,
    id<MTLBuffer> input_buffer,
    id<MTLBuffer> output_buffer,
    uint32_t M,
    uint32_t out_dim,
    uint32_t in_dim,
    uint32_t group_size,
    int batched,
    uint32_t iterations) {
    const double start_ms = monotonic_ms();
    for (uint32_t iteration = 0u; iteration < iterations; ++iteration) {
        id<MTLCommandBuffer> command = [queue commandBuffer];
        if (batched) {
            id<MTLComputeCommandEncoder> encoder =
                [command computeCommandEncoder];
            [encoder setComputePipelineState:bucket_pipeline];
            [encoder setBuffer:weight_buffer offset:0 atIndex:0];
            [encoder setBuffer:scale_buffer offset:0 atIndex:1];
            [encoder setBuffer:input_buffer offset:0 atIndex:2];
            [encoder setBuffer:output_buffer offset:0 atIndex:3];
            [encoder setBytes:&M length:sizeof(M) atIndex:4];
            [encoder setBytes:&out_dim length:sizeof(out_dim) atIndex:5];
            [encoder setBytes:&in_dim length:sizeof(in_dim) atIndex:6];
            [encoder setBytes:&group_size length:sizeof(group_size) atIndex:7];
            [encoder dispatchThreadgroups:
                MTLSizeMake((out_dim + 7u) / 8u, M, 1u)
                threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
            [encoder endEncoding];
        } else {
            for (uint32_t token = 0u; token < M; ++token) {
                id<MTLComputeCommandEncoder> encoder =
                    [command computeCommandEncoder];
                [encoder setComputePipelineState:scalar_pipeline];
                [encoder setBuffer:weight_buffer offset:0 atIndex:0];
                [encoder setBuffer:scale_buffer offset:0 atIndex:1];
                [encoder setBuffer:input_buffer
                            offset:(NSUInteger)token * in_dim * sizeof(float)
                           atIndex:2];
                [encoder setBuffer:output_buffer
                            offset:(NSUInteger)token * out_dim * sizeof(float)
                           atIndex:3];
                [encoder setBytes:&out_dim length:sizeof(out_dim) atIndex:4];
                [encoder setBytes:&in_dim length:sizeof(in_dim) atIndex:5];
                [encoder setBytes:&group_size length:sizeof(group_size) atIndex:6];
                [encoder dispatchThreadgroups:
                    MTLSizeMake((out_dim + 7u) / 8u, 1u, 1u)
                    threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
                [encoder endEncoding];
            }
        }
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) return INFINITY;
    }
    return (monotonic_ms() - start_ms) / (double)iterations;
}

static void test_bucket_mxfp4_down(void) {
    printf("test_bucket_mxfp4_down:\n");
    const uint32_t in_dim = 64u;
    const uint32_t out_dim = 17u;
    const uint32_t group_size = 32u;
    const uint32_t packed_cols = in_dim / 8u;
    const uint32_t num_groups = in_dim / group_size;
    const size_t weight_words = (size_t)out_dim * packed_cols;
    const size_t scale_count = (size_t)out_dim * num_groups;
    uint32_t *weights = malloc(weight_words * sizeof(uint32_t));
    uint8_t *scales = malloc(scale_count);
    for (size_t word = 0u; word < weight_words; ++word) {
        uint32_t packed = 0u;
        for (uint32_t nibble = 0u; nibble < 8u; ++nibble) {
            packed |= (uint32_t)((word * 11u + nibble * 7u + 3u) & 15u)
                << (nibble * 4u);
        }
        weights[word] = packed;
    }
    for (size_t index = 0u; index < scale_count; ++index) {
        scales[index] = (uint8_t)(122u + index % 6u);
    }

    id<MTLComputePipelineState> pipeline = mkpipe("bucket_mxfp4_down");
    const uint32_t bucket_counts[] = {1u, 2u, 3u, 4u};
    for (size_t case_index = 0u;
         case_index < sizeof(bucket_counts) / sizeof(bucket_counts[0]);
         ++case_index) {
        const uint32_t M = bucket_counts[case_index];
        const size_t input_count = (size_t)M * in_dim;
        const size_t output_count = (size_t)M * out_dim;
        const size_t guard_count = 32u;
        float *input = malloc(input_count * sizeof(float));
        float *output = malloc((output_count + guard_count) * sizeof(float));
        float *reference = malloc(output_count * sizeof(float));
        for (size_t index = 0u; index < input_count; ++index) {
            input[index] = ((float)((int)(index % 23u) - 11)) * 0.015625f;
        }
        for (size_t index = 0u; index < output_count + guard_count; ++index) {
            output[index] = -34567.0f;
        }

        for (uint32_t token = 0u; token < M; ++token) {
            for (uint32_t row = 0u; row < out_dim; ++row) {
                float accumulator = 0.0f;
                for (uint32_t group = 0u; group < num_groups; ++group) {
                    const float scale = e8m0_scale(
                        scales[(size_t)row * num_groups + group]);
                    for (uint32_t packed_index = 0u;
                         packed_index < group_size / 8u;
                         ++packed_index) {
                        const uint32_t word = weights[
                            row * packed_cols +
                            group * (group_size / 8u) + packed_index];
                        const uint32_t column =
                            group * group_size + packed_index * 8u;
                        for (uint32_t nibble = 0u; nibble < 8u; ++nibble) {
                            accumulator += MXFP4_LUT[
                                (word >> (nibble * 4u)) & 15u] * scale *
                                input[(size_t)token * in_dim + column + nibble];
                        }
                    }
                }
                reference[(size_t)token * out_dim + row] =
                    round_to_bf16(accumulator);
            }
        }

        id<MTLBuffer> weight_buffer =
            buf(weights, weight_words * sizeof(uint32_t));
        id<MTLBuffer> scale_buffer = buf(scales, scale_count);
        id<MTLBuffer> input_buffer = buf(input, input_count * sizeof(float));
        id<MTLBuffer> output_buffer =
            buf(output, (output_count + guard_count) * sizeof(float));
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:weight_buffer offset:0 atIndex:0];
        [encoder setBuffer:scale_buffer offset:0 atIndex:1];
        [encoder setBuffer:input_buffer offset:0 atIndex:2];
        [encoder setBuffer:output_buffer offset:0 atIndex:3];
        [encoder setBytes:&M length:sizeof(M) atIndex:4];
        [encoder setBytes:&out_dim length:sizeof(out_dim) atIndex:5];
        [encoder setBytes:&in_dim length:sizeof(in_dim) atIndex:6];
        [encoder setBytes:&group_size length:sizeof(group_size) atIndex:7];
        [encoder dispatchThreadgroups:
            MTLSizeMake((out_dim + 7u) / 8u, M + 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];

        float *actual = (float *)[output_buffer contents];
        float max_diff = 0.0f;
        int finite = command.status == MTLCommandBufferStatusCompleted;
        for (size_t index = 0u; index < output_count; ++index) {
            finite = finite && isfinite(actual[index]);
            const float diff = fabsf(actual[index] - reference[index]);
            if (diff > max_diff) max_diff = diff;
        }
        for (size_t index = output_count;
             index < output_count + guard_count;
             ++index) {
            finite = finite && actual[index] == -34567.0f;
        }
        char label[80];
        snprintf(label, sizeof(label), "bucket_mxfp4_down M=%u", M);
        check(label, max_diff, 2e-3f);
        printf("  %-28s dispatches=1 vs %u  %s\n", "finite/tail/dispatch",
               M, finite ? "OK" : "FAIL");
        if (!finite) g_fail = 1;
        free(input);
        free(output);
        free(reference);
    }
    // Representative target-shape warm timing. M=1 compares the new geometry
    // to the established scalar down kernel; M=4 compares one dispatch against
    // four scalar encoders in one command buffer.
    {
        const uint32_t perf_M = 4u;
        const uint32_t perf_in = 2048u;
        const uint32_t perf_out = 4096u;
        const uint32_t perf_gs = 32u;
        const size_t perf_words = (size_t)perf_out * (perf_in / 8u);
        const size_t perf_scales = (size_t)perf_out * (perf_in / perf_gs);
        uint32_t *perf_weights = malloc(perf_words * sizeof(uint32_t));
        uint8_t *perf_scale_values = malloc(perf_scales);
        float *perf_input = malloc(
            (size_t)perf_M * perf_in * sizeof(float));
        for (size_t index = 0u; index < perf_words; ++index) {
            perf_weights[index] = (uint32_t)(
                index * UINT64_C(2654435761) + UINT64_C(0x13579bdf));
        }
        for (size_t index = 0u; index < perf_scales; ++index) {
            perf_scale_values[index] = (uint8_t)(123u + index % 5u);
        }
        for (size_t index = 0u; index < (size_t)perf_M * perf_in; ++index) {
            perf_input[index] = ((float)((int)(index % 17u) - 8)) * 0.015625f;
        }
        id<MTLBuffer> perf_weight_buffer =
            buf(perf_weights, perf_words * sizeof(uint32_t));
        id<MTLBuffer> perf_scale_buffer =
            buf(perf_scale_values, perf_scales);
        id<MTLBuffer> perf_input_buffer = buf(
            perf_input, (size_t)perf_M * perf_in * sizeof(float));
        id<MTLBuffer> perf_bucket_output_buffer = bufz(
            (size_t)perf_M * perf_out * sizeof(float));
        id<MTLBuffer> perf_scalar_output_buffer = bufz(
            (size_t)perf_M * perf_out * sizeof(float));
        id<MTLComputePipelineState> scalar_pipeline =
            mkpipe("dequant_matvec_4bit");

        // Target-shape dispatch equivalence: preserve independent outputs from
        // one bucket dispatch and M scalar encoders, then compare all rows.
        const double target_bucket_run = time_mxfp4_down_dispatch(
            pipeline, scalar_pipeline, perf_weight_buffer, perf_scale_buffer,
            perf_input_buffer, perf_bucket_output_buffer, perf_M, perf_out,
            perf_in, perf_gs, 1, 1u);
        const double target_scalar_run = time_mxfp4_down_dispatch(
            pipeline, scalar_pipeline, perf_weight_buffer, perf_scale_buffer,
            perf_input_buffer, perf_scalar_output_buffer, perf_M, perf_out,
            perf_in, perf_gs, 0, 1u);
        float *target_bucket =
            (float *)[perf_bucket_output_buffer contents];
        float *target_scalar =
            (float *)[perf_scalar_output_buffer contents];
        float target_max_diff = 0.0f;
        int target_finite = isfinite(target_bucket_run) &&
            isfinite(target_scalar_run);
        for (size_t index = 0u;
             index < (size_t)perf_M * perf_out;
             ++index) {
            target_finite = target_finite && isfinite(target_bucket[index]) &&
                isfinite(target_scalar[index]);
            const float diff = fabsf(
                target_bucket[index] - target_scalar[index]);
            if (diff > target_max_diff) target_max_diff = diff;
        }
        check("down target dispatch eq", target_max_diff, 1e-7f);
        printf("  %-28s %s\n", "target dispatch finite",
               target_finite ? "OK" : "FAIL");
        if (!target_finite) g_fail = 1;

        // Warm both paths before collecting robust medians. Seven independent
        // samples with three command buffers each suppress one-off queue noise;
        // alternating order avoids systematically favoring either path.
        (void)time_mxfp4_down_dispatch(
            pipeline, scalar_pipeline, perf_weight_buffer, perf_scale_buffer,
            perf_input_buffer, perf_bucket_output_buffer, perf_M, perf_out,
            perf_in, perf_gs, 1, 2u);
        (void)time_mxfp4_down_dispatch(
            pipeline, scalar_pipeline, perf_weight_buffer, perf_scale_buffer,
            perf_input_buffer, perf_scalar_output_buffer, perf_M, perf_out,
            perf_in, perf_gs, 0, 2u);
        enum { perf_samples = 7 };
        double bucket_m1_samples[perf_samples];
        double scalar_m1_samples[perf_samples];
        double bucket_m4_samples[perf_samples];
        double scalar_m4_samples[perf_samples];
        for (size_t sample = 0u; sample < perf_samples; ++sample) {
            if ((sample & 1u) == 0u) {
                bucket_m1_samples[sample] = time_mxfp4_down_dispatch(
                    pipeline, scalar_pipeline, perf_weight_buffer,
                    perf_scale_buffer, perf_input_buffer,
                    perf_bucket_output_buffer, 1u, perf_out, perf_in, perf_gs,
                    1, 3u);
                scalar_m1_samples[sample] = time_mxfp4_down_dispatch(
                    pipeline, scalar_pipeline, perf_weight_buffer,
                    perf_scale_buffer, perf_input_buffer,
                    perf_scalar_output_buffer, 1u, perf_out, perf_in, perf_gs,
                    0, 3u);
                bucket_m4_samples[sample] = time_mxfp4_down_dispatch(
                    pipeline, scalar_pipeline, perf_weight_buffer,
                    perf_scale_buffer, perf_input_buffer,
                    perf_bucket_output_buffer, perf_M, perf_out, perf_in,
                    perf_gs, 1, 3u);
                scalar_m4_samples[sample] = time_mxfp4_down_dispatch(
                    pipeline, scalar_pipeline, perf_weight_buffer,
                    perf_scale_buffer, perf_input_buffer,
                    perf_scalar_output_buffer, perf_M, perf_out, perf_in,
                    perf_gs, 0, 3u);
            } else {
                scalar_m4_samples[sample] = time_mxfp4_down_dispatch(
                    pipeline, scalar_pipeline, perf_weight_buffer,
                    perf_scale_buffer, perf_input_buffer,
                    perf_scalar_output_buffer, perf_M, perf_out, perf_in,
                    perf_gs, 0, 3u);
                bucket_m4_samples[sample] = time_mxfp4_down_dispatch(
                    pipeline, scalar_pipeline, perf_weight_buffer,
                    perf_scale_buffer, perf_input_buffer,
                    perf_bucket_output_buffer, perf_M, perf_out, perf_in,
                    perf_gs, 1, 3u);
                scalar_m1_samples[sample] = time_mxfp4_down_dispatch(
                    pipeline, scalar_pipeline, perf_weight_buffer,
                    perf_scale_buffer, perf_input_buffer,
                    perf_scalar_output_buffer, 1u, perf_out, perf_in, perf_gs,
                    0, 3u);
                bucket_m1_samples[sample] = time_mxfp4_down_dispatch(
                    pipeline, scalar_pipeline, perf_weight_buffer,
                    perf_scale_buffer, perf_input_buffer,
                    perf_bucket_output_buffer, 1u, perf_out, perf_in, perf_gs,
                    1, 3u);
            }
        }
        const double bucket_m1 = median_double(
            bucket_m1_samples, perf_samples);
        const double scalar_m1 = median_double(
            scalar_m1_samples, perf_samples);
        const double bucket_m4 = median_double(
            bucket_m4_samples, perf_samples);
        const double scalar_m4 = median_double(
            scalar_m4_samples, perf_samples);
        const double m1_regression_budget = 0.15;
        const double m4_speedup_floor = 0.05;
        const int m1_ok = isfinite(bucket_m1) && isfinite(scalar_m1) &&
            bucket_m1 <= scalar_m1 * (1.0 + m1_regression_budget);
        const int m4_ok = isfinite(bucket_m4) && isfinite(scalar_m4) &&
            bucket_m4 <= scalar_m4 * (1.0 - m4_speedup_floor);
        printf("  down target-shape M=1       bucket=%.3fms scalar=%.3fms "
               "budget<=15%%  %s\n",
               bucket_m1, scalar_m1, m1_ok ? "OK" : "FAIL");
        printf("  down target-shape M=4       bucket=%.3fms scalar4=%.3fms "
               "speedup>=5%%  %s\n",
               bucket_m4, scalar_m4, m4_ok ? "OK" : "FAIL");
        if (!m1_ok || !m4_ok) g_fail = 1;
        free(perf_weights);
        free(perf_scale_values);
        free(perf_input);
    }
    free(weights);
    free(scales);
}

// ---- test: Wave-4 stable bucket input gather [B,H] -> [M,H] ----
typedef struct {
    uint32_t token_id;
    uint32_t expert_id;
    uint32_t route_rank;
    float weight;
} NativeActivationPairGPUHost;

static void test_bucket_gather_f32(void) {
    printf("test_bucket_gather_f32:\n");
    static const struct {
        uint32_t batch_tokens;
        uint32_t bucket_count;
        uint32_t hidden_dim;
    } cases[] = {
        {1u, 1u, 17u},
        {2u, 2u, 256u},
        {3u, 3u, 257u},
        {5u, 5u, 513u},
    };
    id<MTLComputePipelineState> pipeline = mkpipe("bucket_gather_f32");

    for (size_t case_index = 0u;
         case_index < sizeof(cases) / sizeof(cases[0]);
         ++case_index) {
        const uint32_t B = cases[case_index].batch_tokens;
        const uint32_t M = cases[case_index].bucket_count;
        const uint32_t H = cases[case_index].hidden_dim;
        const uint32_t start = 1u;
        const size_t source_count = (size_t)B * H;
        const size_t output_count = (size_t)M * H;
        const size_t guard_count = 32u;
        float *source = malloc(source_count * sizeof(float));
        float *output = malloc((output_count + guard_count) * sizeof(float));
        NativeActivationPairGPUHost *pairs = calloc(
            M + 2u, sizeof(NativeActivationPairGPUHost));
        for (uint32_t token = 0u; token < B; ++token) {
            for (uint32_t column = 0u; column < H; ++column) {
                source[(size_t)token * H + column] =
                    (float)(token * 10000u + column);
            }
        }
        for (uint32_t row = 0u; row < M; ++row) {
            pairs[start + row] = (NativeActivationPairGPUHost){
                .token_id = M - 1u - row,
                .expert_id = 7u,
                .route_rank = row % 6u,
                .weight = 0.25f,
            };
        }
        for (size_t index = 0u; index < output_count + guard_count; ++index) {
            output[index] = -12345.0f;
        }

        id<MTLBuffer> source_buffer =
            buf(source, source_count * sizeof(float));
        id<MTLBuffer> pairs_buffer =
            buf(pairs, (M + 2u) * sizeof(NativeActivationPairGPUHost));
        id<MTLBuffer> output_buffer =
            buf(output, (output_count + guard_count) * sizeof(float));
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:source_buffer offset:0 atIndex:0];
        [encoder setBuffer:pairs_buffer offset:0 atIndex:1];
        [encoder setBuffer:output_buffer offset:0 atIndex:2];
        [encoder setBytes:&start length:sizeof(start) atIndex:3];
        [encoder setBytes:&M length:sizeof(M) atIndex:4];
        [encoder setBytes:&B length:sizeof(B) atIndex:5];
        [encoder setBytes:&H length:sizeof(H) atIndex:6];
        // Deliberately over-dispatch one row. All lanes initialize and cross
        // the barrier before the row-tail mask, so the canary must survive.
        [encoder dispatchThreadgroups:MTLSizeMake((H + 255u) / 256u, M + 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];

        float *actual = (float *)[output_buffer contents];
        float max_diff = 0.0f;
        for (uint32_t row = 0u; row < M; ++row) {
            const uint32_t token = M - 1u - row;
            for (uint32_t column = 0u; column < H; ++column) {
                const float expected = source[(size_t)token * H + column];
                const float diff = fabsf(
                    actual[(size_t)row * H + column] - expected);
                if (diff > max_diff) max_diff = diff;
            }
        }
        char label[80];
        snprintf(label, sizeof(label),
                 "bucket_gather_f32 B=%u M=%u H=%u", B, M, H);
        check(label, max_diff, 1e-7f);
        int guard_ok = command.status == MTLCommandBufferStatusCompleted;
        for (size_t index = output_count;
             index < output_count + guard_count;
             ++index) {
            guard_ok = guard_ok && actual[index] == -12345.0f;
        }
        printf("  %-28s %s\n", "row/column tail canary",
               guard_ok ? "OK" : "FAIL");
        if (!guard_ok) g_fail = 1;
        free(source);
        free(output);
        free(pairs);
    }

    // Defensive source-token mask: malformed metadata must not read source[B].
    const uint32_t B = 2u, M = 1u, H = 257u, start = 0u;
    float *source = calloc((size_t)B * H, sizeof(float));
    float *output = malloc((size_t)M * H * sizeof(float));
    NativeActivationPairGPUHost invalid = {
        .token_id = B,
        .expert_id = 9u,
        .route_rank = 0u,
        .weight = 1.0f,
    };
    for (uint32_t column = 0u; column < H; ++column) output[column] = 777.0f;
    id<MTLBuffer> source_buffer = buf(source, (size_t)B * H * sizeof(float));
    id<MTLBuffer> pairs_buffer = buf(&invalid, sizeof(invalid));
    id<MTLBuffer> output_buffer = buf(output, (size_t)M * H * sizeof(float));
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:source_buffer offset:0 atIndex:0];
    [encoder setBuffer:pairs_buffer offset:0 atIndex:1];
    [encoder setBuffer:output_buffer offset:0 atIndex:2];
    [encoder setBytes:&start length:sizeof(start) atIndex:3];
    [encoder setBytes:&M length:sizeof(M) atIndex:4];
    [encoder setBytes:&B length:sizeof(B) atIndex:5];
    [encoder setBytes:&H length:sizeof(H) atIndex:6];
    [encoder dispatchThreadgroups:MTLSizeMake((H + 255u) / 256u, M, 1u)
        threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    float *actual = (float *)[output_buffer contents];
    int invalid_ok = command.status == MTLCommandBufferStatusCompleted;
    for (uint32_t column = 0u; column < H; ++column) {
        invalid_ok = invalid_ok && actual[column] == 777.0f;
    }
    printf("  %-28s %s\n", "invalid token masked", invalid_ok ? "OK" : "FAIL");
    if (!invalid_ok) g_fail = 1;
    free(source);
    free(output);
}

static void test_bucket_combine_f32(void) {
    printf("test_bucket_combine_f32:\n");
    const uint32_t batch_sizes[] = {1u, 2u, 3u, 5u};
    const uint32_t H = 259u;
    const size_t guard_count = 32u;
    id<MTLComputePipelineState> pipeline = mkpipe("bucket_combine_f32");
    for (size_t case_index = 0u;
         case_index < sizeof(batch_sizes) / sizeof(batch_sizes[0]);
         ++case_index) {
        const uint32_t B = batch_sizes[case_index];
        const uint32_t P = B * 6u;
        NativeActivationPairGPUHost *pairs = malloc(
            (size_t)P * sizeof(NativeActivationPairGPUHost));
        uint32_t *identity = malloc((size_t)P * sizeof(uint32_t));
        uint32_t *permuted = malloc((size_t)P * sizeof(uint32_t));
        float *logical = malloc((size_t)P * H * sizeof(float));
        float *bucket_identity = malloc((size_t)P * H * sizeof(float));
        float *bucket_permuted = malloc((size_t)P * H * sizeof(float));
        float *shared = malloc((size_t)B * H * sizeof(float));
        float *expected = malloc((size_t)B * H * sizeof(float));
        float *output_a = malloc(
            ((size_t)B * H + guard_count) * sizeof(float));
        float *output_b = malloc(
            ((size_t)B * H + guard_count) * sizeof(float));

        for (uint32_t token = 0u; token < B; ++token) {
            for (uint32_t rank = 0u; rank < 6u; ++rank) {
                const uint32_t source = token * 6u + rank;
                pairs[source] = (NativeActivationPairGPUHost){
                    .token_id = token,
                    .expert_id = (rank * 17u + token * 3u) % 256u,
                    .route_rank = rank,
                    .weight = (float)(rank + 1u) * 0.03125f +
                        (float)token * 0.001f,
                };
                identity[source] = source;
                permuted[source] = (source * 7u + 3u) % P;
                for (uint32_t column = 0u; column < H; ++column) {
                    logical[(size_t)source * H + column] =
                        (float)((int)((source * 13u + column * 3u) % 37u) - 18) *
                        0.0078125f;
                }
            }
            for (uint32_t column = 0u; column < H; ++column) {
                shared[(size_t)token * H + column] =
                    (float)((int)((token * 11u + column * 5u) % 29u) - 14) *
                    0.00390625f;
            }
        }
        for (uint32_t source = 0u; source < P; ++source) {
            memcpy(bucket_identity + (size_t)identity[source] * H,
                   logical + (size_t)source * H,
                   (size_t)H * sizeof(float));
            memcpy(bucket_permuted + (size_t)permuted[source] * H,
                   logical + (size_t)source * H,
                   (size_t)H * sizeof(float));
        }
        for (uint32_t token = 0u; token < B; ++token) {
            for (uint32_t column = 0u; column < H; ++column) {
                float routed = 0.0f;
                for (uint32_t rank = 0u; rank < 6u; ++rank) {
                    const uint32_t source = token * 6u + rank;
                    routed += logical[(size_t)source * H + column] *
                        pairs[source].weight;
                }
                expected[(size_t)token * H + column] = routed +
                    shared[(size_t)token * H + column];
            }
        }
        for (size_t index = 0u; index < (size_t)B * H + guard_count;
             ++index) {
            output_a[index] = -45678.0f;
            output_b[index] = -45678.0f;
        }

        id<MTLBuffer> pairs_buffer = buf(
            pairs, (size_t)P * sizeof(NativeActivationPairGPUHost));
        id<MTLBuffer> identity_buffer = buf(
            identity, (size_t)P * sizeof(uint32_t));
        id<MTLBuffer> permuted_buffer = buf(
            permuted, (size_t)P * sizeof(uint32_t));
        id<MTLBuffer> bucket_identity_buffer = buf(
            bucket_identity, (size_t)P * H * sizeof(float));
        id<MTLBuffer> bucket_permuted_buffer = buf(
            bucket_permuted, (size_t)P * H * sizeof(float));
        id<MTLBuffer> shared_buffer = buf(
            shared, (size_t)B * H * sizeof(float));
        id<MTLBuffer> output_a_buffer = buf(
            output_a, ((size_t)B * H + guard_count) * sizeof(float));
        id<MTLBuffer> output_b_buffer = buf(
            output_b, ((size_t)B * H + guard_count) * sizeof(float));
        id<MTLCommandBuffer> command = [queue commandBuffer];
        for (uint32_t variant = 0u; variant < 2u; ++variant) {
            id<MTLComputeCommandEncoder> encoder =
                [command computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:variant == 0u
                    ? bucket_identity_buffer : bucket_permuted_buffer
                         offset:0 atIndex:0];
            [encoder setBuffer:pairs_buffer offset:0 atIndex:1];
            [encoder setBuffer:variant == 0u
                    ? identity_buffer : permuted_buffer
                         offset:0 atIndex:2];
            [encoder setBuffer:shared_buffer offset:0 atIndex:3];
            [encoder setBuffer:variant == 0u
                    ? output_a_buffer : output_b_buffer
                         offset:0 atIndex:4];
            [encoder setBytes:&B length:sizeof(B) atIndex:5];
            [encoder setBytes:&P length:sizeof(P) atIndex:6];
            [encoder setBytes:&H length:sizeof(H) atIndex:7];
            [encoder dispatchThreadgroups:
                MTLSizeMake((H + 255u) / 256u, B + 1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
            [encoder endEncoding];
        }
        [command commit];
        [command waitUntilCompleted];

        float *actual_a = (float *)[output_a_buffer contents];
        float *actual_b = (float *)[output_b_buffer contents];
        float max_reference = 0.0f;
        float max_permutation = 0.0f;
        int valid = command.status == MTLCommandBufferStatusCompleted;
        for (size_t index = 0u; index < (size_t)B * H; ++index) {
            valid = valid && isfinite(actual_a[index]) &&
                isfinite(actual_b[index]);
            const float reference_diff = fmaxf(
                fabsf(actual_a[index] - expected[index]),
                fabsf(actual_b[index] - expected[index]));
            const float permutation_diff = fabsf(
                actual_a[index] - actual_b[index]);
            if (reference_diff > max_reference) {
                max_reference = reference_diff;
            }
            if (permutation_diff > max_permutation) {
                max_permutation = permutation_diff;
            }
        }
        for (size_t index = (size_t)B * H;
             index < (size_t)B * H + guard_count;
             ++index) {
            valid = valid && actual_a[index] == -45678.0f &&
                actual_b[index] == -45678.0f;
        }
        char reference_label[80];
        char permutation_label[80];
        snprintf(reference_label, sizeof(reference_label),
                 "bucket_combine B=%u CPU", B);
        snprintf(permutation_label, sizeof(permutation_label),
                 "bucket_combine B=%u permute", B);
        check(reference_label, max_reference, 1e-6f);
        check(permutation_label, max_permutation, 1e-7f);
        printf("  %-28s %s\n", "finite/tail canary",
               valid ? "OK" : "FAIL");
        if (!valid) g_fail = 1;

        free(pairs);
        free(identity);
        free(permuted);
        free(logical);
        free(bucket_identity);
        free(bucket_permuted);
        free(shared);
        free(expected);
        free(output_a);
        free(output_b);
    }
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
    test_shared_expert_batch_chain();
    test_dequant_int8_e8m0();
    test_fused_gate_up_swiglu_int8_e8m0();
    test_bucket_mxfp4_gate_up_swiglu();
    test_bucket_mxfp4_down();
    test_bucket_gather_f32();
    test_bucket_combine_f32();

    printf(g_fail ? "\nRESULT: KERNEL TESTS FAILED\n" : "\nRESULT: ALL KERNEL TESTS PASSED\n");
    return g_fail;
}
