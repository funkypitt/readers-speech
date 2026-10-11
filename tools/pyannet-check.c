/* Host check of pyannet.c: runs the network over one window of a 16 kHz mono s16 wav and writes
 * the frames × 7 log-probabilities as raw float32, printing the frame count and the time.
 *
 *     pyannet-check model.bin audio.wav OFFSET_SAMPLES N_SAMPLES out.f32
 *
 * Compared with onnxruntime by tools/pyannet-check.py. */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../src/main/cpp/pyannet.h"

static void * slurp(const char * path, size_t * n) {
    FILE * f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    void * buf = malloc((size_t) len);
    if (!buf || fread(buf, 1, (size_t) len, f) != (size_t) len) { fclose(f); free(buf); return NULL; }
    fclose(f);
    *n = (size_t) len;
    return buf;
}

int main(int argc, char ** argv) {
    if (argc != 6) { fprintf(stderr, "usage: %s model.bin audio.wav offset n out.f32\n", argv[0]); return 2; }
    size_t mn, wn;
    void * mb = slurp(argv[1], &mn);
    unsigned char * wav = (unsigned char *) slurp(argv[2], &wn);
    if (!mb || !wav) { fprintf(stderr, "cannot read inputs\n"); return 1; }
    pyannet * m = pyannet_load(mb, mn);
    if (!m) { fprintf(stderr, "not a pyannet model\n"); return 1; }
    free(mb);
    /* find the data chunk of the wav */
    size_t pos = 12, data = 0, dlen = 0;
    while (pos + 8 <= wn) {
        unsigned len = wav[pos + 4] | wav[pos + 5] << 8 | wav[pos + 6] << 16 | (unsigned) wav[pos + 7] << 24;
        if (memcmp(wav + pos, "data", 4) == 0) { data = pos + 8; dlen = len; break; }
        pos += 8 + len + (len & 1);
    }
    if (!data) { fprintf(stderr, "no data chunk\n"); return 1; }
    long total = (long) (dlen / 2), off = atol(argv[3]), n = atol(argv[4]);
    if (off < 0 || n <= 0 || off + n > total) { fprintf(stderr, "window out of range (%ld samples)\n", total); return 1; }
    float * pcm = (float *) malloc((size_t) n * sizeof(float));
    for (long i = 0; i < n; i++) {
        const unsigned char * s = wav + data + 2 * (size_t) (off + i);
        pcm[i] = (float) (short) (s[0] | s[1] << 8) / 32768.0f;
    }
    int F = pyannet_frames(m, (int) n);
    float * out = (float *) malloc((size_t) F * PYANNET_CLASSES * sizeof(float) + 4);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int frames = 0;
    int rc = pyannet_run(m, pcm, (int) n, out, &frames);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    if (rc != 0) { fprintf(stderr, "run failed\n"); return 1; }
    double ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    FILE * o = fopen(argv[5], "wb");
    fwrite(out, sizeof(float), (size_t) frames * PYANNET_CLASSES, o);
    fclose(o);
    printf("%d %.1f\n", frames, ms);
    pyannet_free(m);
    free(out); free(pcm); free(wav);
    return 0;
}
