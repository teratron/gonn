# Сравнение вариантов Fluent API

## Вариант 1: Builder с методами цепочки

### ✅ Преимущества

- **Самый читаемый синтаксис** - естественный поток
- **Простая реализация** - понятная структура
- **Хорошая IDE поддержка** - автодополнение всех методов
- **Легко расширяемый** - просто добавить новые методы

### ❌ Недостатки

- Менее гибкий при переиспользовании конфигураций
- Сложнее создавать динамические конфигурации

### 📝 Пример использования

```go
nn := NewBuilder[float32]().
    Input(2).
    Dense(4, activation.SIGMOID).
Dense(4, activation.ReLU).
    Output(1, activation.SIGMOID).
    WithLearningRate(0.3).
    WithLoss(loss.MSE).
    MustCompile()
```

### 💡 Когда использовать

- Для большинства простых задач
- Когда важна читаемость кода
- Для примеров и обучения

---

## Вариант 2: Конфигурационный объект

### ✅ Преимущества

- **Переиспользование конфигураций** - легко сохранять и загружать
- **Сериализация** - можно экспортировать в JSON/YAML
- **Модификация** - легко изменять параметры
- **Валидация** - централизованная проверка

### ❌ Недостатки

- Более многословный синтаксис
- Требует больше кода для реализации
- Две стадии: создание конфига → создание сети

### 📝 Пример использования

```go
config := NewConfig[float32]().
    Input(2).
    AddLayer(4, activation.SIGMOID).
    Output(1, activation.SIGMOID).
    LearningRate(0.3).
    Build()

// Переиспользование
nn1, _ := CreateFromConfig(config)
nn2, _ := CreateFromConfig(config)

// Сохранение
json, _ := config.ToJSON()
```

### 💡 Когда использовать

- Для экспериментов с гиперпараметрами
- Когда нужно сохранять конфигурации
- Для создания множества похожих сетей
- В production коде с валидацией

---

## Вариант 3: Функциональные опции

### ✅ Преимущества

- **Максимальная гибкость** - легко комбинировать опции
- **Расширяемость** - легко добавлять новые опции без breaking changes
- **Композиция** - можно создавать higher-order функции
- **Чистый API** - одна функция с вариативными параметрами

### ❌ Недостатки

- Сложнее для новичков
- Менее очевидный порядок вызова
- Сложнее отлаживать

### 📝 Пример использования

```go
// Простое использование
nn := MustNewNetwork[float32](
    WithInput[float32](2),
    WithHiddenLayer[float32](4, activation.SIGMOID),
    WithOutput[float32](1, activation.SIGMOID),
    WithLearningRate[float32](0.3),
)

// С комбинированными опциями
nn := MustNewNetwork[float32](
    WithInput[float32](784),
DeepNetwork[float32](128, 3, activation.ReLU),
    WithOutput[float32](10, activation.SOFTMAX),
    StandardSetup[float32](0.001),
)

// Переиспользование опций
commonOpts := []Option[float32]{
    WithLearningRate[float32](0.001),
    WithLoss[float32](loss.MSE),
}

nn1 := MustNewNetwork(append(commonOpts, ...)...)
nn2 := MustNewNetwork(append(commonOpts, ...)...)
```

### 💡 Когда использовать

- Для продвинутых пользователей
- Когда нужна максимальная гибкость
- Для создания DSL (Domain Specific Language)
- Для библиотек, которые будут расширяться

---

## 🎯 Мои рекомендации

### Оптимальное решение: **Комбинация вариантов 1 и 3**

Реализуйте оба подхода параллельно:

#### Для простых случаев (80% использования)

```go
// Вариант 1 - Builder
nn := nn.NewBuilder[float32]().
    Input(2).
    Dense(4, activation.SIGMOID).
    Output(1, activation.SIGMOID).
    WithLearningRate(0.3).
    MustCompile()
```

#### Для продвинутых случаев (20% использования)

```go
// Вариант 3 - Functional Options
nn := nn.MustNew[float32](
    nn.WithInput[float32](784),
nn.DeepNetwork[float32](128, 3, activation.ReLU),
    nn.WithOutput[float32](10, activation.SOFTMAX),
    nn.StandardSetup[float32](0.001),
    nn.WithEpochCallback[float32](logProgress),
)
```

---

## 📊 Таблица сравнения

| Критерий | Builder | Config | Functional |
|----------|---------|--------|------------|
| Читаемость | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Простота | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| Гибкость | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Расширяемость | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Переиспользование | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Сериализация | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| IDE поддержка | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

---

## 🚀 План внедрения

### Этап 1: Реализуйте Builder (Вариант 1)

1. Создайте `NetworkBuilder` структуру
2. Реализуйте основные методы: `Input()`, `Dense()`, `Output()`, `Compile()`
3. Добавьте валидацию
4. Напишите тесты и примеры

### Этап 2: Добавьте дополнительные методы

1. `WithLearningRate()`, `WithLoss()`, `WithBias()`
2. `WithWeightInit()`, `WithOptimizer()`
3. Методы для регуляризации: `L1()`, `L2()`, `Dropout()`

### Этап 3: (Опционально) Добавьте Functional Options

1. Создайте тип `Option[T]`
2. Реализуйте базовые опции
3. Добавьте комбинированные опции (Sequential, DeepNetwork)
4. Создайте пресеты для типовых задач

---

## 💻 Минимальная реализация для старта

```go
package nn

type NetworkBuilder[T utils.Float] struct {
    inputSize    int
    hiddenLayers []HiddenLayer
    outputSize   int
    outputAct    activation.Type
    learningRate T
    lossFunc     loss.Type
}

func NewBuilder[T utils.Float]() *NetworkBuilder[T] {
    return &NetworkBuilder[T]{
        learningRate: T(0.01),
        lossFunc:     loss.MSE,
    }
}

func (b *NetworkBuilder[T]) Input(size int) *NetworkBuilder[T] {
    b.inputSize = size
    return b
}

func (b *NetworkBuilder[T]) Dense(neurons int, act activation.Type) *NetworkBuilder[T] {
    b.hiddenLayers = append(b.hiddenLayers, HiddenLayer{
        Number:     neurons,
        Activation: act,
        Bias:       true,
    })
    return b
}

func (b *NetworkBuilder[T]) Output(size int, act activation.Type) *NetworkBuilder[T] {
    b.outputSize = size
    b.outputAct = act
    return b
}

func (b *NetworkBuilder[T]) WithLearningRate(rate T) *NetworkBuilder[T] {
    b.learningRate = rate
    return b
}

func (b *NetworkBuilder[T]) MustCompile() *NN[T] {
    nn := &NN[T]{
        LearningRate: b.learningRate,
        Loss:         b.lossFunc,
    }
    
    nn.Network.SetInputLayer(b.inputSize)
    if len(b.hiddenLayers) > 0 {
        nn.Network.SetHiddenLayers(b.hiddenLayers...)
    }
    nn.Network.SetOutputLayer(b.outputSize, b.outputAct, true)
    nn.Network.Build("xavier")
    
    return nn
}
```

---

## 🎓 Примеры для документации

### XOR задача

```go
nn := nn.NewBuilder[float32]().
    Input(2).
    Dense(4, activation.SIGMOID).
    Output(1, activation.SIGMOID).
    WithLearningRate(0.3).
    MustCompile()
```

### MNIST классификация

```go
nn := nn.NewBuilder[float32]().
    Input(784).
	Dense(128, activation.ReLU).
Dense(64, activation.ReLU).
    Output(10, activation.SOFTMAX).
    WithLearningRate(0.001).
    MustCompile()
```

### Регрессия

```go
nn := nn.NewBuilder[float64]().
    Input(5).
	Dense(20, activation.ReLU).
Dense(10, activation.ReLU).
Output(1, activation.Linear).
    WithLearningRate(0.01).
    MustCompile()
```
