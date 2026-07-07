#!/bin/sh
# Fetch the three public-domain prose corpora (one per prose style) used by collect_grads.py.
set -e
cd "$(dirname "$0")"
mkdir -p data
for id in 1342 2680 2701; do   # Pride and Prejudice / Meditations / Moby-Dick
  [ -f "data/pg$id.txt" ] || curl -sL "https://www.gutenberg.org/cache/epub/$id/pg$id.txt" -o "data/pg$id.txt"
done
wc -c data/pg*.txt
