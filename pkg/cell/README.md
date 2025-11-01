# Cell - Go реализация нейронных клеток

## Введение

Пакет `cell` предоставляет полную Go реализацию различных типов нейронных клеток для построения нейронных сетей. Реализация следует принципам объектно-ориентированного программирования и использует дженерики для поддержки различных числовых типов (`float32`, `float64`).

## Архитектура

Реализация основана на двух основных интерфейсах:

- `Nucleus` - базовый интерфейс для всех клеток
- `Neuron` - расширенный интерфейс для обучаемых клеток

## Интерфейсы

### Nucleus

```go
type Nucleus[T utils.Float] interface {
    GetValue() T
}
```

Базовый интерфейс, который должны реализовывать все клетки. Предоставляет метод `GetValue()` для получения текущего значения клетки.

### Neuron

```go
type Neuron[T utils.Float] interface {
    Nucleus[T]
    GetMiss() T
    CalculateValue() T
    CalculateWeight(miss T) T
    Forward() T
    Backward(target T) T
}
```

Расширенный интерфейс для клеток, которые могут обучаться. Включает методы для:

- `GetMiss()` - получения ошибки
- `CalculateValue()` - вычисления значения
- `CalculateWeight()` - вычисления веса
- `Forward()` - прямого распространения
- `Backward()` - обратного распространения

## Типы клеток

### Input

Входная клетка, которая содержит ссылку на исходные данные и не обучается.

```go
// Создание входной клетки
inputValue := float32(1.0)
inputCell := cell.NewInputCell[float32](&inputValue)

// Установка значения
inputCell.SetValue(0.5)

// Получение значения
value := inputCell.GetValue()
```

**Особенности:**

- Не обучается (всегда возвращает 0 для весов)
- Может содержать ссылку на исходные данные
- Поддерживает прямое и обратное распространение

### Hidden

Скрытая клетка, которая содержит основную функциональность и исходящие связи.

```go
// Создание скрытой клетки с сигмоидной активацией
hiddenCell := cell.NewHiddenCell[float32](cell.SIGMOID)

// Добавление входящей связи
hiddenCell.core.AddIncomingConnection(inputCell, 0.8)

// Добавление исходящей связи
hiddenCell.AddOutgoingConnection(outputCell, 0.6)

// Прямое распространение
value := hiddenCell.Forward()

// Обратное распространение
weight := hiddenCell.Backward(0.1)
```

**Особенности:**

- Содержит `core` с базовой функциональностью
- Поддерживает множественные входящие и исходящие связи
- Использует функции активации

### OutputCell

Выходная клетка с поддержкой целевых значений для обучения.

```go
// Создание выходной клетки
outputCell := cell.NewOutputCell[float32](cell.SIGMOID)

// Установка целевого значения
outputCell.SetTarget(1.0)

// Прямое распространение
value := outputCell.Forward()

// Проверка точности
isCorrect := outputCell.IsCorrect(0.01)

// Получение ошибки
error := outputCell.GetError()
squaredError := outputCell.GetSquaredError()
```

**Особенности:**

- Поддерживает целевые значения для обучения
- Вычисляет ошибку как разность между целевым и полученным значением
- Предоставляет методы для проверки точности

### BiasCell

Клетка смещения, которая всегда возвращает значение 1.0 и не обучается.

```go
// Создание клетки смещения
biasCell := cell.NewBiasCell[float32]()

// Получение значения (всегда 1.0)
value := biasCell.GetValue()
```

**Особенности:**

- Всегда возвращает значение 1.0
- Не обучается
- Используется для добавления смещения к нейронам

### core

Базовая структура, которая содержит общую функциональность для всех клеток.

```go
// Создание базовой клетки
core := cell.NewCoreCell[float32](cell.RELU)

// Добавление входящих связей
core.AddIncomingConnection(neuron1, 0.5)
core.AddIncomingConnection(neuron2, 0.3)

// Установка смещения
core.SetBias(0.1)

// Вычисление значения
value := core.CalculateValue()

// Получение производной функции активации
derivative := core.GetActivationDerivative()
```

## Функции активации

Пакет поддерживает четыре основные функции активации:

### SIGMOID

```go
σ(x) = 1 / (1 + e^(-x))
```

- Диапазон: (0, 1)
- Применение: бинарная классификация

### RELU

```go
ReLU(x) = max(0, x)
```

- Диапазон: [0, +∞)
- Применение: скрытые слои

### TANH

```go
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```

- Диапазон: (-1, 1)
- Применение: скрытые слои

### LINEAR

```go
Linear(x) = x
```

- Диапазон: (-∞, +∞)
- Применение: регрессия

## Связи между клетками

### Axon

Структура для представления связи между нейронами:

```go
// Создание связи
axon := cell.NewAxon[float32](targetNeuron, 0.8)

// Обновление веса
axon.Delta = 0.05
axon.UpdateWeight(0.1) // learning rate = 0.1
```

## Примеры использования

### Простая нейронная сеть

```go
package main

import (
    "fmt"
    "github.com/teratron/gonn/pkg/cell"
)

func main() {
    // Создаем входные клетки
    input1 := cell.NewInputCell[float32](nil)
    input2 := cell.NewInputCell[float32](nil)
    
    input1.SetValue(1.0)
    input2.SetValue(0.5)
    
    // Создаем скрытую клетку
    hidden := cell.NewHiddenCell[float32](cell.SIGMOID)
    
    // Создаем выходную клетку
    output := cell.NewOutputCell[float32](cell.SIGMOID)
    
    // Устанавливаем связи
    hidden.core.AddIncomingConnection(input1, 0.8)
    hidden.core.AddIncomingConnection(input2, 0.6)
    
    output.core.AddIncomingConnection(hidden, 0.7)
    
    // Прямое распространение
    hidden.Forward()
    outputValue := output.Forward()
    
    fmt.Printf("Выходное значение: %f\n", outputValue)
    
    // Обратное распространение
    output.SetTarget(1.0)
    output.Backward(1.0)
    hidden.Backward(output.GetMiss())
}
```

### Использование различных функций активации

```go
// Создаем клетки с разными функциями активации
sigmoidCell := cell.NewHiddenCell[float32](cell.SIGMOID)
reluCell := cell.NewHiddenCell[float32](cell.RELU)
tanhCell := cell.NewHiddenCell[float32](cell.TANH)
linearCell := cell.NewHiddenCell[float32](cell.LINEAR)

// Устанавливаем значения и вычисляем
for _, cell := range []*cell.Hidden[float32]{sigmoidCell, reluCell, tanhCell, linearCell} {
    cell.core.SetBias(1.0)
    value := cell.Forward()
    fmt.Printf("Значение: %f\n", value)
}
```

## Тестирование

### Запуск всех тестов

```bash
# Запуск тестов для пакета cell
go test ./pkg/cell/...

# Запуск тестов с покрытием
go test -coverprofile=coverage.out ./pkg/cell/...

# Просмотр покрытия
go tool cover -html=coverage.out
```

### Основные тесты

Пакет включает следующие тесты:

- `TestAxonCreation` - тестирование создания связей
- `TestAxonWeightUpdate` - тестирование обновления весов
- `TestActivationModes` - тестирование функций активации
- `TestNeuronInterface` - тестирование интерфейсов
- `TestCoreCell*` - тестирование базовой функциональности
- `TestInputCell*` - тестирование входных клеток
- `TestHiddenCell*` - тестирование скрытых клеток
- `TestOutputCell*` - тестирование выходных клеток
- `TestBiasCell*` - тестирование клеток смещения

### Пример запуска конкретного теста

```bash
# Запуск теста функций активации
go test -v ./pkg/cell/... -run TestActivationModes

# Запуск теста интерфейсов
go test -v ./pkg/cell/... -run TestNeuronInterface
```

## Структура файлов

### Основные файлы

- `cell.go` - определения интерфейсов и базовых структур
- `core.go` - реализация core
- `input.go` - реализация Input
- `hidden.go` - реализация Hidden
- `output.go` - реализация OutputCell
- `bias.go` - реализация BiasCell

### Тестовые файлы

- `cell_test.go` - общие тесты интерфейсов и связей
- `core_test.go` - тесты core
- `input_test.go` - тесты Input
- `hidden_test.go` - тесты Hidden
- `output_test.go` - тесты OutputCell
- `bias_test.go` - тесты BiasCell

## Производительность

Реализация оптимизирована для:

- Минимального использования памяти
- Быстрого выполнения операций
- Эффективного обучения нейронных сетей

### Рекомендации по использованию

1. **Выбор типа данных**: Используйте `float32` для лучшей производительности или `float64` для большей точности
2. **Функции активации**: Выбирайте подходящую функцию активации в зависимости от задачи
3. **Инициализация весов**: Начинайте с небольших случайных значений
4. **Скорость обучения**: Настройте скорость обучения в зависимости от задачи

## Зависимости

Пакет использует:

- `github.com/teratron/gonn/pkg/utils` - для унифицированных числовых типов
- `math` - для математических функций

## Лицензия

См. файл LICENSE в корне проекта.
