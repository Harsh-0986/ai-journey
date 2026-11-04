````
# Day 4 — Optimization Algorithms

**Duration:** 3 hours  
**Level:** Intermediate  
**Focus:** Optimization theory, gradient descent variants, momentum, RMSProp, Adam, and learning rate scheduling.

---

## 1. Objectives

By the end of today, you should be able to:

- Understand convex and non-convex optimization.
- Identify local minima, global minima, and saddle points.
- Understand batch, stochastic, and mini-batch gradient descent.
- Implement gradient descent variants from scratch.
- Understand momentum, RMSProp, and Adam.
- Understand learning rate scheduling.
- Compare optimization algorithms experimentally.
- Explain why Adam is widely used in deep learning.

---

## 2. Concepts to Learn

### 2.1 Optimization in Machine Learning

Optimization is the process of finding model parameters that minimize a loss function.

For example, in linear regression:

```text
Loss = MSE(y_pred, y_true)
````

Our objective is to find the weight and bias that minimize this loss.

Learn:

- Objective function
- Parameters
- Constraints
- Optimization landscape
- Convergence
- Local minima
- Global minima
- Saddle points

### 2.2 Convex vs Non-Convex Functions

A convex function has a bowl-like shape. Any local minimum is also a global minimum.

A non-convex function may contain:

- Multiple local minima.
- Saddle points.
- Flat regions.
- Complex optimization landscapes.

Examples:

```
import numpy as npimport matplotlib.pyplot as pltx = np.linspace(-5, 5, 1000)convex = x**2non_convex = x**4 - 5*x**2 + 4plt.figure(figsize=(10, 5))plt.plot(x, convex, label="Convex")plt.plot(x, non_convex, label="Non-convex")plt.legend()plt.grid()plt.show()
```

Tasks:

1. Identify the minimum of each function.
2. Explain why the first function is convex.
3. Identify the multiple minima in the second function.
4. Research why neural network loss functions are generally non-convex.

## 3. Gradient Descent Variants

### 3.1 Batch Gradient Descent

Batch Gradient Descent calculates gradients using the entire training dataset.

```
1. Calculate predictions for all samples.
2. Calculate the total loss.
3. Calculate gradients using all samples.
4. Update parameters.
5. Repeat.
```

Advantages:

- Stable gradient estimates.
- Consistent updates.

Disadvantages:

- Expensive for large datasets.
- Requires processing the entire dataset for every update.

### 3.2 Stochastic Gradient Descent

SGD updates parameters using one randomly selected training sample at a time.

Advantages:

- Frequent updates.
- Lower computation per update.
- Can escape some shallow local minima due to noisy updates.

Disadvantages:

- Noisy convergence.
- Loss may fluctuate.

### 3.3 Mini-Batch Gradient Descent

Mini-batch gradient descent processes a small group of samples per update.

It combines advantages of batch and stochastic gradient descent.

Common batch sizes:

- 16
- 32
- 64
- 128
- 256

Mini-batch gradient descent is widely used in deep learning.

## 4. Practical Implementation — Build Gradient Descent Variants

Create:

`gradient_descent_variants.ipynb`

### Task 1: Generate a Dataset

```
import numpy as npimport matplotlib.pyplot as pltnp.random.seed(42)def generate_dataset(size=1000):    x = np.random.normal(100, 10, size)    noise = np.random.normal(100, 10, size)    y = 7 * x + 9 - noise    return x, yx, y = generate_dataset()x_mean = np.mean(x)x_std = np.std(x)x = (x - x_mean) / x_std
```

### Task 2: Implement the Loss Function

```
def mse(y_pred, y_true):    return np.mean((y_pred - y_true) ** 2)
```

### Task 3: Implement Gradient Calculation

```
def compute_gradients(x, y, w, b):    y_pred = w * x + b    errors = y_pred - y    dw = 2 * np.mean(errors * x)    db = 2 * np.mean(errors)    return dw, db
```

### Task 4: Implement Batch Gradient Descent

```
def batch_gradient_descent(    x,    y,    learning_rate=0.01,    epochs=100,):    w = 0.0    b = 0.0    losses = []    for epoch in range(epochs):        y_pred = w * x + b        current_loss = mse(y_pred, y)        losses.append(current_loss)        dw, db = compute_gradients(x, y, w, b)        w -= learning_rate * dw        b -= learning_rate * db    return w, b, losses
```

### Task 5: Implement Stochastic Gradient Descent

```
def stochastic_gradient_descent(    x,    y,    learning_rate=0.01,    epochs=100,):    w = 0.0    b = 0.0    losses = []    for epoch in range(epochs):        indices = np.random.permutation(len(x))        for index in indices:            xi = x[index]            yi = y[index]            y_pred = w * xi + b            error = y_pred - yi            dw = 2 * error * xi            db = 2 * error            w -= learning_rate * dw            b -= learning_rate * db        losses.append(mse(w * x + b, y))    return w, b, losses
```

### Task 6: Implement Mini-Batch Gradient Descent

```
def mini_batch_gradient_descent(    x,    y,    learning_rate=0.01,    epochs=100,    batch_size=32,):    w = 0.0    b = 0.0    losses = []    for epoch in range(epochs):        indices = np.random.permutation(len(x))        for start in range(0, len(x), batch_size):            batch_indices = indices[start:start + batch_size]            xb = x[batch_indices]            yb = y[batch_indices]            dw, db = compute_gradients(xb, yb, w, b)            w -= learning_rate * dw            b -= learning_rate * db        losses.append(mse(w * x + b, y))    return w, b, losses
```

### Task 7: Compare the Algorithms

```
w_batch, b_batch, batch_losses = batch_gradient_descent(x, y)w_sgd, b_sgd, sgd_losses = stochastic_gradient_descent(x, y)w_mini, b_mini, mini_losses = mini_batch_gradient_descent(x, y)plt.figure(figsize=(10, 5))plt.plot(batch_losses, label="Batch GD")plt.plot(sgd_losses, label="SGD")plt.plot(mini_losses, label="Mini-Batch GD")plt.xlabel("Epoch")plt.ylabel("MSE")plt.title("Gradient Descent Comparison")plt.legend()plt.grid()plt.show()
```

Experiments:

1. Compare batch sizes of 8, 16, 32, 64, and 128.
2. Compare convergence speed.
3. Compare loss fluctuations.
4. Compare final model parameters.
5. Repeat each experiment with multiple random seeds.

## 5. Advanced Optimizers

### 5.1 Momentum

Momentum uses a moving average of previous gradients to accelerate optimization.

Update equations:

```
velocity = beta * velocity + gradient

parameter = parameter - learning_rate * velocity
```

Where:

- beta controls momentum.
- velocity stores accumulated gradient information.
- gradient is the current gradient.

Typical beta:

```
0.9
```

Implement:

```
def gradient_descent_momentum(    x,    y,    learning_rate=0.01,    epochs=100,    beta=0.9,):    w = 0.0    b = 0.0    vw = 0.0    vb = 0.0    losses = []    for epoch in range(epochs):        dw, db = compute_gradients(x, y, w, b)        vw = beta * vw + dw        vb = beta * vb + db        w -= learning_rate * vw        b -= learning_rate * vb        losses.append(mse(w * x + b, y))    return w, b, losses
```

Experiment: Compare momentum values of 0.5, 0.9, and 0.99.

### 5.2 RMSProp

RMSProp maintains an exponentially weighted average of squared gradients.

Update equations:

```
squared_gradient =
    beta * squared_gradient
    + (1 - beta) * gradient²

parameter =
    parameter
    - learning_rate * gradient / sqrt(squared_gradient + epsilon)
```

Implement:

```
def rmsprop(    x,    y,    learning_rate=0.01,    epochs=100,    beta=0.9,    epsilon=1e-8,):    w = 0.0    b = 0.0    sw = 0.0    sb = 0.0    losses = []    for epoch in range(epochs):        dw, db = compute_gradients(x, y, w, b)        sw = beta * sw + (1 - beta) * dw**2        sb = beta * sb + (1 - beta) * db**2        w -= learning_rate * dw / (np.sqrt(sw) + epsilon)        b -= learning_rate * db / (np.sqrt(sb) + epsilon)        losses.append(mse(w * x + b, y))    return w, b, losses
```

Experiment:

- Try beta = 0.9.
- Try beta = 0.99.
- Try different learning rates.
- Compare convergence with momentum.

### 5.3 Adam Optimizer

Adam combines momentum and RMSProp.

It maintains:

- First moment: Exponential moving average of gradients.
- Second moment: Exponential moving average of squared gradients.

Update equations:

```
m = beta1 * m + (1 - beta1) * gradient

v = beta2 * v + (1 - beta2) * gradient²
```

Bias correction:

```
m_hat = m / (1 - beta1^t)

v_hat = v / (1 - beta2^t)
```

Parameter update:

```
parameter =
    parameter
    - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
```

Typical hyperparameters:

| Parameter     | Default |
| ------------- | ------- |
| Learning rate | 0.001   |
| Beta 1        | 0.9     |
| Beta 2        | 0.999   |
| Epsilon       | 1e-8    |

Implement Adam:

```
def adam(    x,    y,    learning_rate=0.001,    epochs=100,    beta1=0.9,    beta2=0.999,    epsilon=1e-8,):    w = 0.0    b = 0.0    mw = 0.0    mb = 0.0    vw = 0.0    vb = 0.0    losses = []    for t in range(1, epochs + 1):        dw, db = compute_gradients(x, y, w, b)        mw = beta1 * mw + (1 - beta1) * dw        mb = beta1 * mb + (1 - beta1) * db        vw = beta2 * vw + (1 - beta2) * dw**2        vb = beta2 * vb + (1 - beta2) * db**2        mw_hat = mw / (1 - beta1**t)        mb_hat = mb / (1 - beta1**t)        vw_hat = vw / (1 - beta2**t)        vb_hat = vb / (1 - beta2**t)
```

Important: Adam's default learning rate is not guaranteed to work for every dataset. Experiment rather than assuming one optimizer is always superior.

## 6. Learning Rate Scheduling

A learning rate schedule changes the learning rate during training.

Common strategies:

### A. Step Decay

Reduce the learning rate after a fixed number of epochs.

### B. Exponential Decay

Gradually decrease the learning rate exponentially.

### C. Cosine Annealing

Decrease the learning rate following a cosine curve.

### Practical exercise

Implement exponential decay:

```
def exponential_decay(initial_lr, epoch, decay_rate=0.01):    return initial_lr * np.exp(-decay_rate * epoch)
```

Visualize:

```
epochs = np.arange(1000)learning_rates = [    exponential_decay(0.1, epoch)    for epoch in epochs]plt.plot(epochs, learning_rates)plt.xlabel("Epoch")plt.ylabel("Learning Rate")plt.title("Exponential Learning Rate Decay")plt.grid()plt.show()
```

Implement step decay and cosine annealing as additional exercises.

## 7. Independent Challenge — Optimizer Benchmark

Build a reusable optimizer benchmarking script.

Compare:

1. Batch Gradient Descent.
2. Stochastic Gradient Descent.
3. Mini-Batch Gradient Descent.
4. Momentum.
5. RMSProp.
6. Adam.

Use the same dataset and initial parameters.

Record:

| Optimizer     | Final Loss | Epochs | Learning Rate | Converged |
| ------------- | ---------- | ------ | ------------- | --------- |
| Batch GD      |            |        |               |           |
| SGD           |            |        |               |           |
| Mini-Batch GD |            |        |               |           |
| Momentum      |            |        |               |           |
| RMSProp       |            |        |               |           |
| Adam          |            |        |               |           |

Investigate:

- Which optimizer converges fastest?
- Which optimizer produces the smoothest loss curve?
- Which optimizer is sensitive to learning rate?
- Does the fastest optimizer always produce the lowest final loss?
- How does batch size affect convergence?

Bonus: Implement all optimizers using a common interface so they can be swapped without modifying the training loop.

## 8. Knowledge Check

Answer without looking up the answers.

1. What is optimization in machine learning?
2. What is the difference between convex and non-convex functions?
3. What is a saddle point?
4. Explain batch gradient descent.
5. Explain stochastic gradient descent.
6. What is mini-batch gradient descent?
7. Why is mini-batch gradient descent commonly used in deep learning?
8. What problem does momentum address?
9. How does RMSProp adapt learning rates?
10. What are the first and second moments in Adam?
11. Why does Adam require bias correction?
12. What is the role of epsilon in Adam?
13. What is learning rate scheduling?
14. Can Adam perform worse than SGD?
15. Why should optimizers be benchmarked rather than selected blindly?

## 9. Deliverables

```
day-04/
│
├── README.md
│
├── notebooks/
│   ├── optimization_landscape.ipynb
│   ├── gradient_descent_variants.ipynb
│   ├── advanced_optimizers.ipynb
│   └── learning_rate_scheduling.ipynb
│
├── src/
│   ├── optimizers.py
│   ├── scheduler.py
│   └── benchmark.py
│
├── experiments/
│   ├── optimizer_comparison.png
│   ├── learning_rate_comparison.png
│   └── convergence_comparison.png
│
└── notes.md
```

## 10. Completion Criteria

- Understand convex and non-convex optimization.
- Understand local minima, global minima, and saddle points.
- Implement batch gradient descent.
- Implement stochastic gradient descent.
- Implement mini-batch gradient descent.
- Implement momentum.
- Implement RMSProp.
- Implement Adam.
- Implement at least two learning rate schedules.
- Compare all optimizers experimentally.
- Document the results.
- Complete the knowledge check.

## 11. Final Reflection

Write your answers in `notes.md`.

1. Why are multiple optimization algorithms necessary?
2. What makes Adam different from ordinary gradient descent?
3. Why is momentum useful?
4. What did you observe when changing batch sizes?
5. Which optimizer performed best in your experiments, and why?

Today's key takeaway: Gradient descent tells us how to optimize parameters. Modern optimizers improve the process by adapting updates using gradient history and parameter-specific information.

## Day 5 Preview — Linear Algebra for Deep Learning

Tomorrow, you will explore:

- Vector spaces.
- Matrix operations.
- Matrix multiplication.
- Transpose and inverse.
- Identity matrices.
- Determinants.
- Eigenvalues and eigenvectors.
- Norms and distances.
- Vectorization with NumPy.
- Matrix representation of linear regression.

You will implement linear regression using matrix operations instead of scalar operations and prepare for neural network computations.

```
