# Graph Report - D:\Projects\src\github.com\teratron\gonn  (2026-04-27)

## Corpus Check
- 54 files · ~9,170 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 190 nodes · 198 edges · 25 communities detected
- Extraction: 68% EXTRACTED · 32% INFERRED · 0% AMBIGUOUS · INFERRED: 64 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]

## God Nodes (most connected - your core abstractions)
1. `Loss()` - 18 edges
2. `Activation()` - 10 edges
3. `Derivative()` - 9 edges
4. `NN[T]` - 6 edges
5. `main()` - 5 edges
6. `bundle[T, _]` - 5 edges
7. `Network[T]` - 5 edges
8. `Dense[T]` - 5 edges
9. `NewDense()` - 4 edges
10. `NewOutput()` - 4 edges

## Surprising Connections (you probably didn't know these)
- `Activation()` --calls--> `elishActivation()`  [INFERRED]
  D:\Projects\src\github.com\teratron\gonn\pkg\activation\activation.go → D:\Projects\src\github.com\teratron\gonn\pkg\activation\elish.go
- `Activation()` --calls--> `eluActivation()`  [INFERRED]
  D:\Projects\src\github.com\teratron\gonn\pkg\activation\activation.go → D:\Projects\src\github.com\teratron\gonn\pkg\activation\elu.go
- `Activation()` --calls--> `linearActivation()`  [INFERRED]
  D:\Projects\src\github.com\teratron\gonn\pkg\activation\activation.go → D:\Projects\src\github.com\teratron\gonn\pkg\activation\linear.go
- `Activation()` --calls--> `reluActivation()`  [INFERRED]
  D:\Projects\src\github.com\teratron\gonn\pkg\activation\activation.go → D:\Projects\src\github.com\teratron\gonn\pkg\activation\relu.go
- `Activation()` --calls--> `seluActivation()`  [INFERRED]
  D:\Projects\src\github.com\teratron\gonn\pkg\activation\activation.go → D:\Projects\src\github.com\teratron\gonn\pkg\activation\selu.go

## Communities

### Community 0 - "Community 0"
Cohesion: 0.06
Nodes (19): arctanLoss(), avgLoss(), bceLoss(), catHingeLoss(), cceLossSingle(), hingeLoss(), huberLoss(), huberLossWithDelta() (+11 more)

### Community 1 - "Community 1"
Cohesion: 0.07
Nodes (22): Activation(), Derivative(), Function, Sigmoid, Type, elishActivation(), elishDerivative(), eluActivation() (+14 more)

### Community 2 - "Community 2"
Cohesion: 0.13
Nodes (4): Dense[T], Output[T], bundle[T, _], Network[T]

### Community 3 - "Community 3"
Cohesion: 0.12
Nodes (10): TestActivationTypeString(), Dense, Output, NewDense(), Dense, Dense[T], Output, TestLossTypeString() (+2 more)

### Community 4 - "Community 4"
Cohesion: 0.2
Nodes (3): Axon[T], Input[T], bundle[T, S]

### Community 5 - "Community 5"
Cohesion: 0.31
Nodes (2): main(), NN[T]

### Community 6 - "Community 6"
Cohesion: 0.29
Nodes (4): Input, NewInput(), Input, Input[T]

### Community 7 - "Community 7"
Cohesion: 0.29
Nodes (4): Output[T], init(), New(), NN

### Community 8 - "Community 8"
Cohesion: 0.29
Nodes (4): newBundle(), bundle, Network, New()

### Community 9 - "Community 9"
Cohesion: 0.4
Nodes (3): newBase(), base, base[T, S]

### Community 10 - "Community 10"
Cohesion: 0.4
Nodes (3): core, newCore(), core

### Community 11 - "Community 11"
Cohesion: 0.5
Nodes (2): Axon, Bundle

### Community 12 - "Community 12"
Cohesion: 0.67
Nodes (1): Sigmoid[T]

### Community 13 - "Community 13"
Cohesion: 0.67
Nodes (2): Neuron, Nucleus

### Community 14 - "Community 14"
Cohesion: 0.67
Nodes (1): Bias

### Community 15 - "Community 15"
Cohesion: 0.67
Nodes (1): core[T]

### Community 16 - "Community 16"
Cohesion: 1.0
Nodes (0): 

### Community 17 - "Community 17"
Cohesion: 1.0
Nodes (1): Bias[T]

### Community 18 - "Community 18"
Cohesion: 1.0
Nodes (1): Float

### Community 19 - "Community 19"
Cohesion: 1.0
Nodes (0): 

### Community 20 - "Community 20"
Cohesion: 1.0
Nodes (0): 

### Community 21 - "Community 21"
Cohesion: 1.0
Nodes (0): 

### Community 22 - "Community 22"
Cohesion: 1.0
Nodes (0): 

### Community 23 - "Community 23"
Cohesion: 1.0
Nodes (0): 

### Community 24 - "Community 24"
Cohesion: 1.0
Nodes (0): 

## Knowledge Gaps
- **25 isolated node(s):** `Function`, `Sigmoid`, `base`, `base[T, S]`, `core` (+20 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 16`** (2 nodes): `cosineLossVector()`, `cosine.go`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 17`** (2 nodes): `Bias[T]`, `.GetValue()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 18`** (2 nodes): `float.go`, `Float`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 19`** (1 nodes): `propagation.go`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 20`** (1 nodes): `builder.go`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 21`** (1 nodes): `config.go`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 22`** (1 nodes): `query.go`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 23`** (1 nodes): `train.go`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 24`** (1 nodes): `verify.go`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `CalculateTotalLoss()` connect `Community 0` to `Community 2`?**
  _High betweenness centrality (0.134) - this node is a cross-community bridge._
- **Why does `Type` connect `Community 3` to `Community 0`?**
  _High betweenness centrality (0.103) - this node is a cross-community bridge._
- **Are the 16 inferred relationships involving `Loss()` (e.g. with `mseLoss()` and `maeLoss()`) actually correct?**
  _`Loss()` has 16 INFERRED edges - model-reasoned connections that need verification._
- **Are the 9 inferred relationships involving `Activation()` (e.g. with `elishActivation()` and `eluActivation()`) actually correct?**
  _`Activation()` has 9 INFERRED edges - model-reasoned connections that need verification._
- **Are the 8 inferred relationships involving `Derivative()` (e.g. with `elishDerivative()` and `eluDerivative()`) actually correct?**
  _`Derivative()` has 8 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `main()` (e.g. with `.Output()` and `.Dense()`) actually correct?**
  _`main()` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Function`, `Sigmoid`, `base` to the rest of the system?**
  _25 weakly-connected nodes found - possible documentation gaps or missing edges._