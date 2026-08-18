# Sample report

- Checkpoint: `artifacts/checkpoints/final.pt`
- Training step: 50
- Model: d_model=64, n_layers=2, n_heads=4, d_ff=256, context_length=64, vocab_size=512

## Evaluation

- Loss: 5.5126
- Perplexity: 247.7989
- Tokens evaluated: 537280
- Batches: 1050

## Generations

### Sample 1

Decoding settings: `mode=sampling, temperature=0.8, top_k=40, seed=1337, max_new_tokens=40`

**Prompt:** 'Alice was beginning to'

**Generated:**

```
Alice was beginning toas_ asath“hanthewasbeptoandwasased ccstantoangdwasaningallgsobe_edp,saiding
```

### Sample 2

Decoding settings: `mode=sampling, temperature=0.8, top_k=40, seed=1338, max_new_tokens=40`

**Prompt:** 'The Queen of Hearts'

**Generated:**

```
The Queen of Heartsofin“instnhsaidit pistgasanlyhofsaidandinaniinshehhcc p
aaallthe 
h
```

### Sample 3

Decoding settings: `mode=sampling, temperature=0.8, top_k=40, seed=1339, max_new_tokens=40`

**Prompt:** 'Down the rabbit-hole'

**Generated:**

```
Down the rabbit-hole.pl__totocatwasating,”said_edtoanananshectoofAlice
Alice,  nshetoasn
beandst
```
