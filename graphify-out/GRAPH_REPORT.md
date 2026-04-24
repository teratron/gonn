# Graph Report - .  (2026-04-24)

## Corpus Check
- Corpus is ~7,628 words - fits in a single context window. You may not need a graph.

## Summary
- 292 nodes · 341 edges · 24 communities detected
- Extraction: 77% EXTRACTED · 23% INFERRED · 0% AMBIGUOUS · INFERRED: 78 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Loss Function Impls|Loss Function Impls]]
- [[_COMMUNITY_Activation Core API|Activation Core API]]
- [[_COMMUNITY_Network & Neuron Core|Network & Neuron Core]]
- [[_COMMUNITY_Project Docs & Guidelines|Project Docs & Guidelines]]
- [[_COMMUNITY_Loss Functions Reference|Loss Functions Reference]]
- [[_COMMUNITY_Dense Layer & Tests|Dense Layer & Tests]]
- [[_COMMUNITY_Output Layer & NN Facade|Output Layer & NN Facade]]
- [[_COMMUNITY_Activation Functions Reference|Activation Functions Reference]]
- [[_COMMUNITY_Perceptron Example & NN API|Perceptron Example & NN API]]
- [[_COMMUNITY_Input Layer|Input Layer]]
- [[_COMMUNITY_Network Bundle|Network Bundle]]
- [[_COMMUNITY_Core LayerNeuron|Core Layer/Neuron]]
- [[_COMMUNITY_Base Layer|Base Layer]]
- [[_COMMUNITY_Axon|Axon]]
- [[_COMMUNITY_Linear Activation|Linear Activation]]
- [[_COMMUNITY_Categorical Cross-Entropy|Categorical Cross-Entropy]]
- [[_COMMUNITY_Huber Loss|Huber Loss]]
- [[_COMMUNITY_Neuron Root|Neuron Root]]
- [[_COMMUNITY_Bias Cell|Bias Cell]]
- [[_COMMUNITY_Sigmoid Activation|Sigmoid Activation]]
- [[_COMMUNITY_Cosine Loss|Cosine Loss]]
- [[_COMMUNITY_Core Cell Accessors|Core Cell Accessors]]
- [[_COMMUNITY_Float Utility|Float Utility]]
- [[_COMMUNITY_Bias Accessors|Bias Accessors]]

## God Nodes (most connected - your core abstractions)
1. `Loss()` - 19 edges
2. `Loss Functions` - 19 edges
3. `Activation Functions` - 12 edges
4. `Activation()` - 11 edges
5. `Derivative()` - 10 edges
6. `Go Development Rules` - 8 edges
7. `GoNN Library` - 7 edges
8. `main()` - 6 edges
9. `NewDense()` - 6 edges
10. `NewOutput()` - 6 edges

## Surprising Connections (you probably didn't know these)
- `Swish` --semantically_similar_to--> `Sigmoid`  [INFERRED] [semantically similar]
  pkg/activation/README.md → C:\Projects\src\github.com\teratron\gonn\pkg\activation\sigmoid.go
- `Activation()` --calls--> `linearActivation()`  [INFERRED]
  C:\Projects\src\github.com\teratron\gonn\pkg\activation\activation.go → C:\Projects\src\github.com\teratron\gonn\pkg\activation\linear.go
- `Activation()` --calls--> `SigmoidActivation()`  [INFERRED]
  C:\Projects\src\github.com\teratron\gonn\pkg\activation\activation.go → C:\Projects\src\github.com\teratron\gonn\pkg\activation\sigmoid.go
- `Derivative()` --calls--> `SigmoidDerivative()`  [INFERRED]
  C:\Projects\src\github.com\teratron\gonn\pkg\activation\activation.go → C:\Projects\src\github.com\teratron\gonn\pkg\activation\sigmoid.go
- `Activation Functions` --references--> `Sigmoid`  [EXTRACTED]
  pkg/activation/README.md → C:\Projects\src\github.com\teratron\gonn\pkg\activation\sigmoid.go

## Hyperedges (group relationships)
- **Neural Network Core Components** — activation_readme, loss_readme, cell_readme, readme_gonn [INFERRED 0.80]
- **Go Quality Discipline Toolchain** — agents_gofmt, agents_golangci_lint, agents_testing_quality, agents_pprof [EXTRACTED 0.90]
- **ReLU Family Activation Functions** — activation_relu, activation_leaky_relu, activation_elu, activation_selu, activation_elish [INFERRED 0.85]

## Communities

### Community 0 - "Loss Function Impls"
Cohesion: 0.05
Nodes (16): arctanLoss(), avgLoss(), bceLoss(), catHingeLoss(), hingeLoss(), kldLoss(), logCoshLoss(), CalculateTotalLoss() (+8 more)

### Community 1 - "Activation Core API"
Cohesion: 0.09
Nodes (18): Activation(), Derivative(), Function, Type, elishActivation(), elishDerivative(), eluActivation(), eluDerivative() (+10 more)

### Community 2 - "Network & Neuron Core"
Cohesion: 0.08
Nodes (7): Axon[T], Dense[T], Input[T], Output[T], bundle[T, _], bundle[T, S], Network[T]

### Community 3 - "Project Docs & Guidelines"
Cohesion: 0.11
Nodes (20): Completion Protocol, Concurrency (Goroutines & Channels), ECS Design Principles, Effective Go (Citation), Go Development Rules, gofmt, golangci-lint, Integration & Architecture (+12 more)

### Community 4 - "Loss Functions Reference"
Cohesion: 0.14
Nodes (18): Arctan Error, AVG (Average Error), BCE (Binary Cross-Entropy), Categorical Hinge Loss, CCE (Categorical Cross-Entropy), Cosine Similarity/Distance, Hinge Loss, Huber Loss (+10 more)

### Community 5 - "Dense Layer & Tests"
Cohesion: 0.15
Nodes (7): TestActivationTypeString(), Dense, NewDense(), Dense, Dense[T], TestLossTypeString(), Type

### Community 6 - "Output Layer & NN Facade"
Cohesion: 0.15
Nodes (7): Output, Output, Output[T], init(), New(), NN, NewOutput()

### Community 7 - "Activation Functions Reference"
Cohesion: 0.2
Nodes (14): ELISH (ELU + Sigmoid), ELU, Leaky ReLU, Linear/Identity, Activation Functions, ReLU, SELU, Sigmoid (+6 more)

### Community 8 - "Perceptron Example & NN API"
Cohesion: 0.27
Nodes (2): main(), NN[T]

### Community 9 - "Input Layer"
Cohesion: 0.28
Nodes (4): Input, NewInput(), Input, Input[T]

### Community 10 - "Network Bundle"
Cohesion: 0.32
Nodes (4): newBundle(), bundle, Network, New()

### Community 11 - "Core Layer/Neuron"
Cohesion: 0.38
Nodes (3): core, newCore(), core

### Community 12 - "Base Layer"
Cohesion: 0.4
Nodes (3): newBase(), base, base[T, S]

### Community 13 - "Axon"
Cohesion: 0.6
Nodes (3): Axon, Bundle, New()

### Community 14 - "Linear Activation"
Cohesion: 0.67
Nodes (2): linearActivation(), linearDerivative()

### Community 15 - "Categorical Cross-Entropy"
Cohesion: 0.67
Nodes (2): cceLoss(), cceLossSingle()

### Community 16 - "Huber Loss"
Cohesion: 0.83
Nodes (2): huberLoss(), huberLossWithDelta()

### Community 17 - "Neuron Root"
Cohesion: 0.67
Nodes (2): Neuron, Nucleus

### Community 18 - "Bias Cell"
Cohesion: 0.67
Nodes (2): _NewBias(), Bias

### Community 19 - "Sigmoid Activation"
Cohesion: 0.67
Nodes (1): Sigmoid[T]

### Community 20 - "Cosine Loss"
Cohesion: 0.67
Nodes (1): cosineLossVector()

### Community 21 - "Core Cell Accessors"
Cohesion: 0.67
Nodes (1): core[T]

### Community 22 - "Float Utility"
Cohesion: 0.67
Nodes (1): Float

### Community 23 - "Bias Accessors"
Cohesion: 1.0
Nodes (1): Bias[T]

## Knowledge Gaps
- **27 isolated node(s):** `base[T, S]`, `Dense[T]`, `Input[T]`, `Output[T]`, `Bias[T]` (+22 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Perceptron Example & NN API`** (10 nodes): `main.go`, `main.go`, `main()`, `NN[T]`, `.Dense()`, `.Input()`, `.Output()`, `.Query()`, `.Train()`, `.Verify()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Linear Activation`** (4 nodes): `linear.go`, `linearActivation()`, `linearDerivative()`, `linear.go`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Categorical Cross-Entropy`** (4 nodes): `cce.go`, `cceLoss()`, `cceLossSingle()`, `cce.go`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Huber Loss`** (4 nodes): `huber.go`, `huberLoss()`, `huberLossWithDelta()`, `huber.go`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Neuron Root`** (4 nodes): `neuron.go`, `Neuron`, `Nucleus`, `neuron.go`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Bias Cell`** (4 nodes): `_NewBias()`, `bias.go`, `Bias`, `bias.go`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Sigmoid Activation`** (3 nodes): `Sigmoid[T]`, `.Activation()`, `.Derivative()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Cosine Loss`** (3 nodes): `cosine.go`, `cosineLossVector()`, `cosine.go`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Core Cell Accessors`** (3 nodes): `core[T]`, `.GetValue()`, `.SetValue()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Float Utility`** (3 nodes): `float.go`, `float.go`, `Float`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Bias Accessors`** (2 nodes): `Bias[T]`, `.GetValue()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Loss()` connect `Loss Function Impls` to `Huber Loss`, `Categorical Cross-Entropy`?**
  _High betweenness centrality (0.119) - this node is a cross-community bridge._
- **Why does `CalculateTotalLoss()` connect `Loss Function Impls` to `Network & Neuron Core`?**
  _High betweenness centrality (0.084) - this node is a cross-community bridge._
- **Why does `Type` connect `Dense Layer & Tests` to `Loss Function Impls`?**
  _High betweenness centrality (0.072) - this node is a cross-community bridge._
- **Are the 16 inferred relationships involving `Loss()` (e.g. with `mseLoss()` and `maeLoss()`) actually correct?**
  _`Loss()` has 16 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `Loss Functions` (e.g. with `GoNN Library` and `Perceptron Neural Network Example`) actually correct?**
  _`Loss Functions` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `Activation Functions` (e.g. with `GoNN Library` and `Perceptron Neural Network Example`) actually correct?**
  _`Activation Functions` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 9 inferred relationships involving `Activation()` (e.g. with `elishActivation()` and `eluActivation()`) actually correct?**
  _`Activation()` has 9 INFERRED edges - model-reasoned connections that need verification._