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

**What is and isn't claimed.** The reader is given the clean drive `u_t` directly, not a feature scraped off
the pixels — perception is not the object of study here; the watcher's *memory* is. The film shows that the
input side can be a real, watchable modality; the read still comes from the raw streams, and it is correct.
A natural next step is to let the watcher's input be a feature extracted from the frames (correlated, not
white) and prewhiten inside the reader before the same read — the honest "watches the pixels" version.

Reproduce:

```
pip install numpy matplotlib
cd watched && python visualize.py      # writes watched.png
python watched.py                      # prints the recovered order + pole error
```
