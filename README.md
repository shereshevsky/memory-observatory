# The Memory Observatory

Temporal knowledge graphs, No. 4. The Narrative Observatory's dial applied to a
team conversation, with a query-time memory retrieval (T-Mem, arXiv 2610.12874)
replayed on it: seeds, episode, forest, answer. Companion to the article
"Your Agent Memory Doesn't Need a Graph. It Needs a Clock."

Live: https://shereshevsky.github.io/memory-observatory/

Previous instrument: https://shereshevsky.github.io/temporal-narrative-evolution/

## How to read it

- **Time is the dial.** Four months of a five-person team's chat run clockwise
  from the top. Every message is a grain on the rim; the inner ring is messages
  per day.
- **Connection is gravity.** Each entity is a star placed at the angle of its
  first mention and pulled toward the centre by its relations. This star graph is
  what a write-time memory system maintains.
- **The needle is the query.** When the dial completes, a question fires and the
  instrument replays the retrieval: BM25 seeds, DBSCAN over their timestamps, one
  labelling pass over the window (Key, Support, Irrelevant) with Update and
  Expand links, then the forest serialised into the answer.
- **The cost is the point.** Nothing was paid when the messages arrived. A toggle
  keeps the write-time graph lit during the query for comparison.

## Files

- `build_memory_observatory.py` generates the synthetic conversation, runs a toy
  BM25 and DBSCAN over it, applies the annotated labels and links, and writes
  `memory_kg.json` and `docs/index.html` (the template with the data inlined).
- `template.html` is the page. Edit it, not `docs/index.html`.
- `docs/index.html` is the built page, served by GitHub Pages.
- `memory_kg.json` is the data, in the previous instrument's shape (metadata,
  entities, relations) plus `days`, `messages` and `query`.

## Build

    python3 build_memory_observatory.py

Then commit `docs/index.html` and `memory_kg.json` with the change. The page
needs no server: the data is inline and the only external resource is the
Playfair Display font from Google Fonts, with a serif fallback.

## Title image

`docs/index.html#poster` hides the chrome, centres the dial and freezes the
replay at the forest stage. Rendered with headless Chrome at 2x:

    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
      --headless=new --disable-gpu --no-first-run --hide-scrollbars \
      --user-data-dir=/tmp/chrome-poster --force-device-scale-factor=2 \
      --window-size=1400,700 --timeout=12000 \
      --screenshot="$PWD/cover-observatory.png" "file://$PWD/docs/index.html#poster"

## Data

The conversation is synthetic and narratively anchored: five people, 1 January
to 30 April 2026, about 2,100 messages, with episodes for a search incident,
release 2.3, hiring and the auth refactor the query asks about. The labels that
T-Mem's model would assign are annotated in the build script; the seeds and the
window are computed. No model is called at build time or in the page.
