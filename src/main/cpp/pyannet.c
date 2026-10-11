/* pyannote's segmentation-3.0 network in plain C — see pyannet.h and tools/export-pyannet.py.
 *
 * The graph, read node by node from sherpa-onnx's export of the model (opset 13):
 *
 *   x [1, 1, T]                                 16 kHz PCM
 *   InstanceNormalization (scale, bias, eps 1e-5)  over the waveform itself
 *   Conv 1→80, kernel 251, stride 10, no bias   the sinc filters, baked as constants
 *   Abs; MaxPool 3/3; InstanceNormalization(80, affine); LeakyRelu 0.01
 *   Conv 80→60, kernel 5 (+bias); MaxPool 3/3; InstanceNormalization(60); LeakyRelu
 *   Conv 60→60, kernel 5 (+bias); MaxPool 3/3; InstanceNormalization(60); LeakyRelu
 *   four LSTM layers, bidirectional, hidden 128, gates i,o,f,c, sigmoid/tanh/tanh,
 *     the two directions' outputs concatenated (forward first) between layers
 *   MatMul 256→128 + bias; LeakyRelu; MatMul 128→128 + bias; LeakyRelu; MatMul 128→7 + bias
 *   LogSoftmax                                  y [1, F, 7]
 *
 * For T = 160000 the lengths are 15975 → 5325 → 5321 → 1773 → 1769 → 589 frames, each
 * 270 samples apart, with a receptive field of 991 samples.
 *
 * The two heavy parts — the sinc convolution and the LSTM stack — go through one small
 * register-tiled kernel (gemm4/gemv1: four rows × sixteen columns of accumulators, written so
 * that -O3 vectorises it without reassociating floating point); the smaller convolutions are in
 * the "axpy" form, out[t] += w · in[t] over contiguous t. The LSTM's input projection is done
 * for all frames at once, then the recurrence frame by frame. Measured on one desktop core:
 * about 75 ms per ten-second window with clang, the NDK's compiler. */
#include "pyannet.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef PYANNET_PROFILE
#include <stdio.h>
#include <time.h>
static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
#define STAGE(name) do { double _t = now_ms(); fprintf(stderr, "  %-10s %6.1f ms\n", name, _t - prof_t); prof_t = _t; } while (0)
#else
#define STAGE(name) do { } while (0)
#endif

#define MAGIC "PYAN"
#define VERSION 1
#define GATES 4

struct lstm_dir {
    const float * wt;    /* [in][4H]  */
    const float * rt;    /* [H][4H]   */
    const float * b;     /* [4H]      */
};

struct pyannet {
    int sinc_c, sinc_k, sinc_s;
    int c1, k1, c2, k2, pool;
    int layers, hidden, d0, d1, classes;
    float eps, alpha;
    float * weights;     /* one block; everything below points into it */
    const float * wav_scale, * wav_bias;
    const float * sinc;                       /* [sinc_c][sinc_k] */
    const float * n0_scale, * n0_bias;
    const float * conv1_w, * conv1_b;         /* [c1][sinc_c][k1] */
    const float * n1_scale, * n1_bias;
    const float * conv2_w, * conv2_b;         /* [c2][c1][k2] */
    const float * n2_scale, * n2_bias;
    struct lstm_dir lstm[8][2];               /* [layer][direction] */
    const float * lin0_w, * lin0_b;           /* [2H][d0] */
    const float * lin1_w, * lin1_b;           /* [d0][d1] */
    const float * cls_w, * cls_b;             /* [d1][classes] */
};

static int32_t rd_i32(const uint8_t * p) { return (int32_t) ((uint32_t) p[0] | (uint32_t) p[1] << 8 | (uint32_t) p[2] << 16 | (uint32_t) p[3] << 24); }
static float rd_f32(const uint8_t * p) { uint32_t u = (uint32_t) p[0] | (uint32_t) p[1] << 8 | (uint32_t) p[2] << 16 | (uint32_t) p[3] << 24; float f; memcpy(&f, &u, 4); return f; }

pyannet * pyannet_load(const void * data, size_t n) {
    const uint8_t * p = (const uint8_t *) data;
    const size_t header = 4 + 4 + 13 * 4 + 2 * 4 + 4;
    if (!p || n < header || memcmp(p, MAGIC, 4) != 0 || rd_i32(p + 4) != VERSION) return NULL;
    pyannet * m = (pyannet *) calloc(1, sizeof *m);
    if (!m) return NULL;
    int32_t h[13];
    for (int i = 0; i < 13; i++) h[i] = rd_i32(p + 8 + 4 * i);
    m->sinc_c = h[0]; m->sinc_k = h[1]; m->sinc_s = h[2];
    m->c1 = h[3]; m->k1 = h[4]; m->c2 = h[5]; m->k2 = h[6]; m->pool = h[7];
    m->layers = h[8]; m->hidden = h[9]; m->d0 = h[10]; m->d1 = h[11]; m->classes = h[12];
    m->eps = rd_f32(p + 60); m->alpha = rd_f32(p + 64);
    int32_t count = rd_i32(p + 68);
    if (m->layers < 1 || m->layers > 8 || m->classes != PYANNET_CLASSES || m->pool < 1 || m->sinc_s < 1 || count < 0 || n < header + 4 * (size_t) count
        || m->hidden <= 0 || (GATES * m->hidden) % 16 != 0) { free(m); return NULL; }
    const int H = m->hidden, G = GATES * H;
    size_t need = 2 + (size_t) m->sinc_c * m->sinc_k + 2 * m->sinc_c
                + (size_t) m->c1 * m->sinc_c * m->k1 + m->c1 + 2 * m->c1
                + (size_t) m->c2 * m->c1 * m->k2 + m->c2 + 2 * m->c2;
    for (int l = 0; l < m->layers; l++) need += 2 * ((size_t) (l == 0 ? m->c2 : 2 * H) * G + (size_t) H * G + G);
    need += (size_t) 2 * H * m->d0 + m->d0 + (size_t) m->d0 * m->d1 + m->d1 + (size_t) m->d1 * m->classes + m->classes;
    if ((size_t) count != need) { free(m); return NULL; }
    m->weights = (float *) malloc(need * sizeof(float));
    if (!m->weights) { free(m); return NULL; }
    /* little-endian on every target we have; the memcpy is also the alignment */
    memcpy(m->weights, p + header, need * sizeof(float));
    const float * w = m->weights;
#define TAKE(field, count) do { m->field = w; w += (count); } while (0)
    TAKE(wav_scale, 1); TAKE(wav_bias, 1);
    TAKE(sinc, (size_t) m->sinc_c * m->sinc_k);
    TAKE(n0_scale, m->sinc_c); TAKE(n0_bias, m->sinc_c);
    TAKE(conv1_w, (size_t) m->c1 * m->sinc_c * m->k1); TAKE(conv1_b, m->c1);
    TAKE(n1_scale, m->c1); TAKE(n1_bias, m->c1);
    TAKE(conv2_w, (size_t) m->c2 * m->c1 * m->k2); TAKE(conv2_b, m->c2);
    TAKE(n2_scale, m->c2); TAKE(n2_bias, m->c2);
    for (int l = 0; l < m->layers; l++) for (int d = 0; d < 2; d++) {
        const int in = l == 0 ? m->c2 : 2 * H;
        TAKE(lstm[l][d].wt, (size_t) in * G); TAKE(lstm[l][d].rt, (size_t) H * G); TAKE(lstm[l][d].b, G);
    }
    TAKE(lin0_w, (size_t) 2 * H * m->d0); TAKE(lin0_b, m->d0);
    TAKE(lin1_w, (size_t) m->d0 * m->d1); TAKE(lin1_b, m->d1);
    TAKE(cls_w, (size_t) m->d1 * m->classes); TAKE(cls_b, m->classes);
#undef TAKE
    return m;
}

void pyannet_free(pyannet * m) {
    if (!m) return;
    free(m->weights);
    free(m);
}

/* ── the lengths through the front end ─────────────────────────────────────────────────── */
struct lens { int l1, p1, l2, p2, l3, f; };

static int lengths(const pyannet * m, int n, struct lens * s) {
    if (n < m->sinc_k) return -1;
    s->l1 = (n - m->sinc_k) / m->sinc_s + 1;
    if (s->l1 < m->pool) return -1;
    s->p1 = (s->l1 - m->pool) / m->pool + 1;
    s->l2 = s->p1 - m->k1 + 1;
    if (s->l2 < m->pool) return -1;
    s->p2 = (s->l2 - m->pool) / m->pool + 1;
    s->l3 = s->p2 - m->k2 + 1;
    if (s->l3 < m->pool) return -1;
    s->f = (s->l3 - m->pool) / m->pool + 1;
    return s->f > 0 ? 0 : -1;
}

int pyannet_frames(const pyannet * m, int n) {
    struct lens s;
    return m && lengths(m, n, &s) == 0 ? s.f : 0;
}

/* ── the layers ────────────────────────────────────────────────────────────────────────── */

/* ONNX InstanceNormalization over one channel of length n, in place: scale * (x - mean) / sqrt(var + eps) + bias. */
static void instance_norm(float * x, int n, float scale, float bias, float eps) {
    double sum = 0.0, sq = 0.0;
    for (int i = 0; i < n; i++) sum += x[i];
    const double mean = sum / n;
    for (int i = 0; i < n; i++) { const double d = x[i] - mean; sq += d * d; }
    const float inv = (float) (1.0 / sqrt(sq / n + (double) eps));
    const float a = scale * inv, b = bias - (float) mean * a;
    for (int i = 0; i < n; i++) x[i] = x[i] * a + b;
}

static void leaky_relu(float * x, int n, float alpha) {
    for (int i = 0; i < n; i++) x[i] = x[i] > 0.0f ? x[i] : x[i] * alpha;
}

/* MaxPool kernel = stride = pool, no padding: out[i] = max of in[pool*i .. pool*i+pool). */
static void max_pool(const float * in, int n_out, int pool, float * out) {
    for (int i = 0; i < n_out; i++) {
        const float * s = in + (size_t) i * pool;
        float v = s[0];
        for (int k = 1; k < pool; k++) if (s[k] > v) v = s[k];
        out[i] = v;
    }
}

/* Conv Cin→Cout, kernel K, stride 1, with bias; in is [Cin][len_in], out [Cout][len_out]. */
static void conv1d(const float * restrict in, int cin, int len_in, const float * restrict w, const float * restrict bias, int cout, int K, float * restrict out) {
    const int len_out = len_in - K + 1;
    for (int co = 0; co < cout; co++) {
        float * o = out + (size_t) co * len_out;
        const float b = bias[co];
        for (int t = 0; t < len_out; t++) o[t] = b;
        for (int ci = 0; ci < cin; ci++) {
            const float * wk = w + ((size_t) co * cin + ci) * K;
            const float * src = in + (size_t) ci * len_in;
            for (int k = 0; k < K; k++) {
                const float v = wk[k];
                const float * sk = src + k;
                for (int t = 0; t < len_out; t++) o[t] += v * sk[t];
            }
        }
    }
}

/* ── small matrix kernels ───────────────────────────────────────────────────────────────────
 * o[r][j] += Σ_k x[r][k] · rows[k][j], 16 columns at a time, so that the tile of accumulators
 * lives in registers for the whole k loop (eight AVX2 or sixteen NEON registers for four rows)
 * and each row of weights is read once per four frames. rows[k] is row k of the weight matrix:
 * a table of pointers, so that the sinc layer can present the phases of the waveform as rows.
 * Columns must be a multiple of 16. */
#define TILE 16

static void gemm4(const float * restrict x0, const float * restrict x1, const float * restrict x2, const float * restrict x3, int in,
                  const float * const * restrict rows, int cols, float * restrict o0, float * restrict o1, float * restrict o2, float * restrict o3) {
    for (int j0 = 0; j0 < cols; j0 += TILE) {
        float a0[TILE], a1[TILE], a2[TILE], a3[TILE];
        for (int j = 0; j < TILE; j++) { a0[j] = o0[j0 + j]; a1[j] = o1[j0 + j]; a2[j] = o2[j0 + j]; a3[j] = o3[j0 + j]; }
        for (int k = 0; k < in; k++) {
            const float * restrict wk = rows[k] + j0;
            const float b0 = x0[k], b1 = x1[k], b2 = x2[k], b3 = x3[k];
            for (int j = 0; j < TILE; j++) { a0[j] += b0 * wk[j]; a1[j] += b1 * wk[j]; a2[j] += b2 * wk[j]; a3[j] += b3 * wk[j]; }
        }
        for (int j = 0; j < TILE; j++) { o0[j0 + j] = a0[j]; o1[j0 + j] = a1[j]; o2[j0 + j] = a2[j]; o3[j0 + j] = a3[j]; }
    }
}

static void gemv1(const float * restrict x0, int in, const float * const * restrict rows, int cols, float * restrict o0) {
    for (int j0 = 0; j0 < cols; j0 += TILE) {
        float a0[TILE];
        for (int j = 0; j < TILE; j++) a0[j] = o0[j0 + j];
        for (int k = 0; k < in; k++) {
            const float * restrict wk = rows[k] + j0;
            const float b0 = x0[k];
            for (int j = 0; j < TILE; j++) a0[j] += b0 * wk[j];
        }
        for (int j = 0; j < TILE; j++) o0[j0 + j] = a0[j];
    }
}

/* The sinc layer: Conv 1→C, kernel K, stride S, no bias, then Abs. As a matrix product
 * out[c][t] = Σ_k w[c][k] · X[k][t] with X[k][t] = x[S t + k]: X's row k is phase k mod S of the
 * waveform (xp[p][j] = x[S j + p], zero beyond the end) shifted by k / S — contiguous memory, so
 * the tiled kernel applies as it is. out is [C][l1_pad], l1_pad a multiple of 16, zeroed here. */
static void sinc_conv_abs(const pyannet * m, const float * restrict xp, int phase_len, int l1_pad, const float ** rows, float * restrict out) {
    const int K = m->sinc_k, S = m->sinc_s, C = m->sinc_c;
    for (int k = 0; k < K; k++) rows[k] = xp + (size_t) (k % S) * phase_len + k / S;
    memset(out, 0, (size_t) C * l1_pad * sizeof(float));
    int c = 0;
    for (; c + 4 <= C; c += 4) {
        const float * w = m->sinc + (size_t) c * K;
        float * o = out + (size_t) c * l1_pad;
        gemm4(w, w + K, w + 2 * K, w + 3 * K, K, rows, l1_pad, o, o + l1_pad, o + 2 * l1_pad, o + 3 * l1_pad);
    }
    for (; c < C; c++) gemv1(m->sinc + (size_t) c * K, K, rows, l1_pad, out + (size_t) c * l1_pad);
    for (size_t i = 0; i < (size_t) C * l1_pad; i++) out[i] = fabsf(out[i]);
}

/* exp over a vector, in place, without libm so that the loop vectorises: e = 2^n · p(r) with
 * n = round(x / ln 2), r = x − n ln 2, p a degree-6 polynomial (Cephes expf; relative error
 * about 2e-7, i.e. the float itself). Inputs are clamped to ±87, where the result is 0 or huge
 * but finite, which the sigmoid and tanh below turn into 0 or 1 just as the exact function would. */
static void vexp(float * x, int n) {
    for (int i = 0; i < n; i++) {
        float v = x[i];
        v = v > 87.0f ? 87.0f : (v < -87.0f ? -87.0f : v);
        const float fn = floorf(v * 1.44269504088896341f + 0.5f);
        const float r = v - fn * 0.693145751953125f - fn * 1.428606765330187e-06f;
        float p = 1.9875691500e-4f;
        p = p * r + 1.3981999507e-3f;
        p = p * r + 8.3334519073e-3f;
        p = p * r + 4.1665795894e-2f;
        p = p * r + 1.6666665459e-1f;
        p = p * r + 5.0000001201e-1f;
        p = p * r * r + r + 1.0f;
        const int32_t e = ((int32_t) fn + 127) << 23;
        float scale;
        memcpy(&scale, &e, 4);
        x[i] = p * scale;
    }
}

/* One direction of one LSTM layer. x: [F][in]; writes h for every frame into y[t][y_stride] at y_off.
 * Gates along the 4H: input, output, forget, cell (ONNX order). g is scratch [F][4H]; e is scratch [4H]. */
static void lstm_dir(const struct lstm_dir * L, const float * x, int F, int in, int H, int backward, float * g, float * y, int y_stride, int y_off,
                     float * h, float * c, float * e, const float ** rows) {
    const int G = GATES * H;
    for (int k = 0; k < in; k++) rows[k] = L->wt + (size_t) k * G;
    /* input projection and bias for all frames at once */
    for (int t = 0; t < F; t++) memcpy(g + (size_t) t * G, L->b, (size_t) G * sizeof(float));
    int t = 0;
    for (; t + 4 <= F; t += 4) {
        const float * x0 = x + (size_t) t * in;
        float * g0 = g + (size_t) t * G;
        gemm4(x0, x0 + in, x0 + 2 * in, x0 + 3 * in, in, rows, G, g0, g0 + G, g0 + 2 * G, g0 + 3 * G);
    }
    for (; t < F; t++) gemv1(x + (size_t) t * in, in, rows, G, g + (size_t) t * G);
    for (int k = 0; k < H; k++) rows[k] = L->rt + (size_t) k * G;
    /* the recurrence: i = σ(gi), o = σ(go), f = σ(gf), c = f·c + i·tanh(gc), h = o·tanh(c) */
    memset(h, 0, (size_t) H * sizeof(float));
    memset(c, 0, (size_t) H * sizeof(float));
    for (int step = 0; step < F; step++) {
        const int ti = backward ? F - 1 - step : step;
        float * gt = g + (size_t) ti * G;
        gemv1(h, H, rows, G, gt);
        /* e = exp(-g) for the three sigmoids, exp(-2 g) for the cell's tanh = 2σ(2g) − 1 */
        for (int j = 0; j < 3 * H; j++) e[j] = -gt[j];
        for (int j = 3 * H; j < G; j++) e[j] = -2.0f * gt[j];
        vexp(e, G);
        const float * ei = e, * eo = e + H, * ef = e + 2 * H, * ec = e + 3 * H;
        for (int j = 0; j < H; j++) {
            const float i_ = 1.0f / (1.0f + ei[j]), f_ = 1.0f / (1.0f + ef[j]);
            const float c_ = 2.0f / (1.0f + ec[j]) - 1.0f;
            c[j] = f_ * c[j] + i_ * c_;
        }
        /* tanh(c) = 2σ(2c) − 1, through the same exp; the sigmoid of o is still in eo */
        float * tc = e + 3 * H;
        for (int j = 0; j < H; j++) tc[j] = -2.0f * c[j];
        vexp(tc, H);
        float * out = y + (size_t) ti * y_stride + y_off;
        for (int j = 0; j < H; j++) {
            h[j] = (1.0f / (1.0f + eo[j])) * (2.0f / (1.0f + tc[j]) - 1.0f);
            out[j] = h[j];
        }
    }
}

/* y[F][out] = x[F][in] · w[in][out] + b, then leaky relu when alpha >= 0. */
static void linear(const float * restrict x, int F, int in, const float * restrict w, const float * restrict b, int out, float alpha, float * restrict y) {
    for (int t = 0; t < F; t++) {
        float * o = y + (size_t) t * out;
        memcpy(o, b, (size_t) out * sizeof(float));
        const float * xt = x + (size_t) t * in;
        for (int k = 0; k < in; k++) {
            const float a = xt[k];
            const float * wk = w + (size_t) k * out;
            for (int j = 0; j < out; j++) o[j] += a * wk[j];
        }
        if (alpha >= 0.0f) leaky_relu(o, out, alpha);
    }
}

static void log_softmax(float * x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += expf(x[i] - mx);
    const float lse = mx + logf(sum);
    for (int i = 0; i < n; i++) x[i] -= lse;
}

int pyannet_run(const pyannet * m, const float * pcm, int n, float * out, int * frames) {
    struct lens s;
    if (frames) *frames = 0;
    if (!m || !pcm || !out || lengths(m, n, &s) != 0) return -1;
    const int H = m->hidden, G = GATES * H, F = s.f;
    const int l1_pad = (s.l1 + TILE - 1) / TILE * TILE;
    const int phase_len = l1_pad + m->sinc_k / m->sinc_s + 1;
    const int n_rows = m->sinc_k > 2 * H ? m->sinc_k : 2 * H;
    /* one block of scratch */
    const size_t sz_x = (size_t) n, sz_xp = (size_t) m->sinc_s * phase_len,
                 sz_c1 = (size_t) m->sinc_c * l1_pad, sz_p1 = (size_t) m->sinc_c * s.p1,
                 sz_c2 = (size_t) m->c1 * s.l2, sz_p2 = (size_t) m->c1 * s.p2,
                 sz_c3 = (size_t) m->c2 * s.l3, sz_p3 = (size_t) m->c2 * F,
                 sz_seq = (size_t) F * 2 * H, sz_g = (size_t) F * G,
                 sz_lin = (size_t) F * (m->d0 > m->d1 ? m->d0 : m->d1);
    const size_t total = sz_x + sz_xp + sz_c1 + sz_p1 + sz_c2 + sz_p2 + sz_c3 + sz_p3 + 2 * sz_seq + sz_g + 2 * sz_lin + 2 * H + G;
    float * mem = (float *) malloc(total * sizeof(float));
    const float ** rows = (const float **) malloc((size_t) n_rows * sizeof(float *));
    if (!mem || !rows) { free(mem); free(rows); return -1; }
#ifdef PYANNET_PROFILE
    double prof_t = now_ms();
#endif
    float * x = mem, * xp = x + sz_x, * c1 = xp + sz_xp, * p1 = c1 + sz_c1, * c2 = p1 + sz_p1, * p2 = c2 + sz_c2,
          * c3 = p2 + sz_p2, * p3 = c3 + sz_c3, * seqa = p3 + sz_p3, * seqb = seqa + sz_seq, * g = seqb + sz_seq,
          * lin0 = g + sz_g, * lin1 = lin0 + sz_lin, * h = lin1 + sz_lin, * c = h + H, * e = c + H;

    /* the waveform, normalised, then split into its stride phases */
    memcpy(x, pcm, (size_t) n * sizeof(float));
    instance_norm(x, n, m->wav_scale[0], m->wav_bias[0], m->eps);
    for (int p = 0; p < m->sinc_s; p++) {
        float * dst = xp + (size_t) p * phase_len;
        for (int j = 0; j < phase_len; j++) {
            const long idx = (long) m->sinc_s * j + p;
            dst[j] = idx < n ? x[idx] : 0.0f;
        }
    }
    STAGE("norm");
    /* SincNet */
    sinc_conv_abs(m, xp, phase_len, l1_pad, rows, c1);
    STAGE("sinc");
    for (int ch = 0; ch < m->sinc_c; ch++) {
        float * o = p1 + (size_t) ch * s.p1;
        max_pool(c1 + (size_t) ch * l1_pad, s.p1, m->pool, o);
        instance_norm(o, s.p1, m->n0_scale[ch], m->n0_bias[ch], m->eps);
        leaky_relu(o, s.p1, m->alpha);
    }
    STAGE("pool0");
    conv1d(p1, m->sinc_c, s.p1, m->conv1_w, m->conv1_b, m->c1, m->k1, c2);
    STAGE("conv1");
    for (int ch = 0; ch < m->c1; ch++) {
        float * o = p2 + (size_t) ch * s.p2;
        max_pool(c2 + (size_t) ch * s.l2, s.p2, m->pool, o);
        instance_norm(o, s.p2, m->n1_scale[ch], m->n1_bias[ch], m->eps);
        leaky_relu(o, s.p2, m->alpha);
    }
    conv1d(p2, m->c1, s.p2, m->conv2_w, m->conv2_b, m->c2, m->k2, c3);
    STAGE("conv2");
    for (int ch = 0; ch < m->c2; ch++) {
        float * o = p3 + (size_t) ch * F;
        max_pool(c3 + (size_t) ch * s.l3, F, m->pool, o);
        instance_norm(o, F, m->n2_scale[ch], m->n2_bias[ch], m->eps);
        leaky_relu(o, F, m->alpha);
    }
    /* [C][F] → [F][C] for the recurrent layers */
    for (int ch = 0; ch < m->c2; ch++) for (int t = 0; t < F; t++) seqa[(size_t) t * m->c2 + ch] = p3[(size_t) ch * F + t];
    /* the LSTM stack: each layer reads [F][in] and writes [F][2H], forward half then backward half */
    float * cur = seqa, * nxt = seqb;
    for (int l = 0; l < m->layers; l++) {
        const int in = l == 0 ? m->c2 : 2 * H;
        lstm_dir(&m->lstm[l][0], cur, F, in, H, 0, g, nxt, 2 * H, 0, h, c, e, rows);
        lstm_dir(&m->lstm[l][1], cur, F, in, H, 1, g, nxt, 2 * H, H, h, c, e, rows);
        float * t = cur; cur = nxt; nxt = t;
        STAGE("lstm");
    }
    /* the head */
    linear(cur, F, 2 * H, m->lin0_w, m->lin0_b, m->d0, m->alpha, lin0);
    linear(lin0, F, m->d0, m->lin1_w, m->lin1_b, m->d1, m->alpha, lin1);
    linear(lin1, F, m->d1, m->cls_w, m->cls_b, m->classes, -1.0f, out);
    for (int t = 0; t < F; t++) log_softmax(out + (size_t) t * m->classes, m->classes);
    STAGE("head");
    free(mem);
    free(rows);
    if (frames) *frames = F;
    return 0;
}
