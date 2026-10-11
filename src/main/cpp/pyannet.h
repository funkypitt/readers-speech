/* pyannote's segmentation-3.0 network (PyanNet: SincNet front end, four bidirectional LSTM
 * layers, two linear layers, a seven-class log-softmax), written out in plain C so that the
 * module needs no inference engine for it. Weights come from the file written by
 * tools/export-pyannet.py. Plain C99, single-threaded, no dependencies beyond libm. */
#ifndef PYANNET_H
#define PYANNET_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct pyannet pyannet;

/** The model from the bytes of the .bin file (copied; the buffer may go). NULL when they are not it. */
pyannet * pyannet_load(const void * data, size_t n);
void pyannet_free(pyannet * m);

/** Classes per frame: none, A, B, C, A+B, A+C, B+C. */
#define PYANNET_CLASSES 7

/** How many frames [pyannet_run] writes for n samples (0 when n is shorter than the receptive field). */
int pyannet_frames(const pyannet * m, int n);

/** Runs the network over n samples of 16 kHz mono PCM in [-1, 1]; writes frames × 7 log-probabilities
 * to out (at least pyannet_frames(m, n) * 7 floats) and the frame count to *frames. 0 on success,
 * -1 when memory is short or the sound too short. */
int pyannet_run(const pyannet * m, const float * pcm, int n, float * out, int * frames);

#ifdef __cplusplus
}
#endif

#endif
