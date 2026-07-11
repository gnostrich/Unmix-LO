# watched — recognizable modality in, the model's memory read from the streams alone

Every earlier I/O picture in this repo showed abstract number-streams, because at the layer the reader
operates on, the data *is* number-streams. This demo puts a human-recognizable modality on the input side
without changing what the reader does.

**The pipeline (left to right in `watched.png`):**

- **INPUT — a bouncing-ball film.** A real physics scene (`orbit/physics.py`, unchanged) driven frame by
  frame by a white signal `u_t`. Each frame is one white kick rendered as forces on the balls, so the input
  is a modality your eyes parse — you can watch it — while `u_t` itself stays white.
- **WATCHER — the model in the middle.** A linear recurrence `x_{t+1}=A x_t + B u_t`, `y_t = C x_t + noise`,
  with a **planted** memory order `r` and known poles `eig(A)`. It is fed the same drive and emits `y_t`.
  Its hidden state `x_t` is shown greyed out — it is never handed to the reader.
- **READER — the verdict.** `io_trace/stream_trace.read_trace`, unmodified. It sees **only** the two streams
  `(u_t, y_t)` — never the weights, never `x_t` — and recovers how much memory the watcher carries (the
  order) and where its poles sit. Because we planted the poles, the answer is checkable: it lands them
  (order 4/4, pole-match error ~0.016).

**What is and isn't claimed here.** In `watched.py` the reader is given the clean drive `u_t` directly, not a
feature scraped off the pixels — perception is not the object of study, the watcher's *memory* is. That first
picture shows the input side can be a real, watchable modality while the read still comes from the raw
streams and is correct.

## The nuanced version — the watcher reads the pixels (`nuanced.py`, `watched_nuanced.png`)

Let the watcher's input be a feature actually extracted from the rendered frames (a fixed random projection
of the pixels). Now the input is **correlated** — the physics is smooth, so consecutive frames are similar
(lag-1 ρ ≈ 0.55). The fit-free Markov estimate `h_k = (1/T) Σ_t y_{t+k} f_t^T` is then the true kernel
*convolved with the input's own autocorrelation* `R_ff`, so read naively you recover the input's color, not
the watcher:

- **naive** read of `(f, y)`: order 3 (planted 4), pole error ≈ 0.47 — poles pulled toward the input's spectrum.
- **deconvolved** read: order 4, pole error ≈ 0.000 — the planted poles come back.

The deconvolution (`correlated_read.py`) is just the least-squares regression of `y_t` on the stacked lags
`[f_t, …, f_{t-K}]`, which divides out `R_ff`; everything downstream (Hankel → permutation floor →
Ho-Kalman → poles) is reused from `stream_trace` unchanged. Calibration (`python correlated_read.py`):

- **white input control** — deconvolving reader matches the white reader (it breaks nothing).
- **colored input control** (AR(1), a=0.9, known poles) — the naive reader returns the input's pole ≈ 0.9
  (order 2, err 0.75); the deconvolving reader recovers the watcher (err 0.01).

**Honest cost.** The white reader had no free knobs; the deconvolution needs a tiny ridge (default 1e-3,
relative to the normal matrix) to stay conditioned when `R_ff` is near-singular. It is disclosed, kept small,
and does not affect the white or well-excited cases.

Reproduce:

```
pip install numpy matplotlib
cd watched
python visualize.py          # watched.png        — white-drive version (reader handed the clean drive)
python nuanced.py            # watched_nuanced.png — watcher reads the pixels; naive fails, deconv recovers
python correlated_read.py    # calibration: white + colored controls
python watched.py            # prints the recovered order + pole error
```
