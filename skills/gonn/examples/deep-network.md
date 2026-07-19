# Deep Network — 100-Layer Construction

**Goal:** Build a 100-hidden-layer network using bulk constructors and pair
it with an LR scheduler for stable deep training.

## Code — Uniform Repeat (Builder)

```go
package main

import (
    "fmt"
    "log"

    "github.com/teratron/gonn/pkg/activation"
    "github.com/teratron/gonn/pkg/nn"
    "github.com/teratron/gonn/pkg/optimizer"
)

func main() {
    opt := optimizer.NewAdam[float64](0.001)

    // Warm-up for 1000 steps → cosine decay for 9000 steps.
    warmup := optimizer.NewWarmUpLR[float64](0.001, 1000)
    cosine := optimizer.NewCosineAnnealingLR[float64](0.001, 1e-6, 9000)
    chain := optimizer.NewChainScheduler([]optimizer.SchedulerSegment[float64]{
        {Scheduler: warmup, Duration: 1000},
        {Scheduler: cosine, Duration: 9000},
    })
    sched := optimizer.BindScheduler[float64](opt, chain)

    net, err := nn.NewBuilder[float64]().
        Input(784).
        Repeat(100, 256, activation.ReLU, true).  // 100 hidden layers
        Output(10, activation.SOFTMAX, true).
        WithWeightInit(nn.WeightInitHe).           // He init for ReLU stacks
        WithOptimizer(opt).
        WithScheduler(sched).
        Compile()
    if err != nil {
        log.Fatal(err)
    }

    fmt.Printf("Compiled: %d hidden layers\n", len(net.Config().HiddenLayers))
    // Output: Compiled: 100 hidden layers
}
```

## Code — Pattern Block (Options)

```go
// 3-layer bottleneck block repeated 33 times = 99 hidden layers + 1 final = 100.
block := []nn.HiddenLayerSpec[float64]{
    {Size: 512, Activation: activation.ReLU, Bias: true},
    {Size: 256, Activation: activation.ReLU, Bias: true},
    {Size: 512, Activation: activation.ReLU, Bias: true},
}

net, err := nn.New[float64](
    nn.WithInput[float64](784),
    nn.Pattern[float64](block, 33),
    nn.WithHiddenLayer[float64](256, activation.ReLU),  // +1 = 100 total
    nn.WithOutput[float64](10, activation.SOFTMAX),
    nn.WithWeightInit[float64](nn.WeightInitHe),
)
```

## Explanation

1. `Repeat(100, 256, activation.ReLU, true)` — 100 identical 256-ReLU-bias layers in O(N).
2. `WeightInitHe` — required for ReLU stacks to prevent gradient vanishing.
3. `BindScheduler` — connects the chain scheduler to the Adam optimizer so
   each `sched.Step()` call (from the training loop) updates `opt.LearningRate()`.
4. `WithScheduler(sched)` — training loop calls `sched.Step()` per epoch (PerEpoch
   granularity) or per batch (PerStep); the scheduler's Granularity() determines this.

## Anti-patterns

```go
// Wrong: Random init for deep ReLU stacks
.WithWeightInit(nn.WeightInitRandom)  // compile warns; gradient explosion likely

// Wrong: no scheduler for 100+ layer network
// Without LR scheduling, Adam's fixed LR often causes instability in deep nets.

// Wrong: forgetting BindScheduler
sched := optimizer.NewStepLR[float64](0.1, 10, 0.5)
// opt's LR never updates — schedule advances but has no effect on optimizer
nn.WithScheduler[float64](sched)  // must BindScheduler(opt, sched) first
```
