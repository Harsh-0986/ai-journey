
# Day 01 — Mathematical Foundations of Machine Learning

> 90 Days of Machine Learning, Deep Learning & AI  
> Phase 1: Machine Learning Foundations  
> Duration: 3 Hours  
> Difficulty: Intermediate

---

# 1. Learning Objectives

Today's goal is to understand the mathematical foundations behind Machine Learning and implement your first ML algorithm without relying on Scikit-learn.

By the end of today, you should be able to:

- Understand scalars, vectors and matrices.
- Perform vector and matrix operations using NumPy.
- Understand the mathematics behind Linear Regression.
- Understand weights, bias and predictions.
- Understand loss functions and Mean Squared Error.
- Understand derivatives and gradients.
- Implement Gradient Descent from scratch.
- Visualize how a model learns.

**Main project:** Linear Regression From Scratch.

---

# 2. Daily Schedule

| Task | Duration |
|---|---:|
| Linear Algebra Fundamentals | 45 min |
| NumPy Implementation | 30 min |
| Linear Regression Theory | 30 min |
| Implement Gradient Descent | 45 min |
| Experiments and Visualization | 15 min |
| Documentation and Revision | 15 min |
| **Total** | **180 min** |

---

# 3. Task 1 — Linear Algebra Fundamentals

**Duration:** 45 minutes

Linear Algebra is one of the most important mathematical foundations of Machine Learning.

Neural networks, regression models and almost every modern ML architecture use vectors and matrices to represent and process data.

## 3.1 Scalars

A scalar is a single numerical value.

Examples:

```python
x = 5
learning_rate = 0.01
temperature = 32.5
```

## 3.2 Vectors

A vector is an ordered collection of numerical values.

For example:

$$
v = \begin{bmatrix}1\\2\\3\end{bmatrix}
$$

In Machine Learning, a vector can represent the features of a single observation.

```python
import numpy as np

v = np.array([1, 2, 3])

print(v.shape)
print(v.ndim)
```

Understand:

- What is a vector?
- What is its dimension?
- What is the difference between a row vector and column vector?
- How do vector addition and scalar multiplication work?

## 3.3 Matrices

A matrix is a two-dimensional arrangement of numbers.

$$
A =
\begin{bmatrix}
1 & 2\\
3 & 4
\end{bmatrix}
$$

In ML, a dataset is commonly represented as a matrix.

- Rows represent observations.
- Columns represent features.

```python
A = np.array([
    [1, 2],
    [3, 4]
])

print(A.shape)
print(A.ndim)
```

Understand:

- Matrix dimensions.
- Matrix addition.
- Matrix multiplication.
- Matrix transpose.
- Identity matrix.

## 3.4 Dot Product

The dot product multiplies corresponding elements of two vectors and sums the results.

$$
A \cdot B = \sum_{i=1}^{n} A_iB_i
$$

Example:

$$
\begin{aligned}
A &= [1,2,3]\\
B &= [4,5,6]
\end{aligned}
$$

$$
A \cdot B = 1(4)+2(5)+3(6)=32
$$

Implementation:

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

result = np.dot(a, b)
print(result)
```

### Why is this important?

A linear model with multiple features uses the dot product to calculate predictions:

$$
\hat{y} = \mathbf{w}^T\mathbf{x}+b
$$

Understand how this connects Linear Algebra with Machine Learning.

## 3.5 Matrix Multiplication

For two matrices:

$$
A_{m \times n} \cdot B_{n \times p}
$$

The resulting matrix has dimensions:

$$
C_{m \times p}
$$

Example:

$$
A =
\begin{bmatrix}
1 & 2\\
3 & 4
\end{bmatrix}
$$

$$
B =
\begin{bmatrix}
5 & 6\\
7 & 8
\end{bmatrix}
$$

$$
AB =
\begin{bmatrix}
19 & 22\\
43 & 50
\end{bmatrix}
$$

## 3.6 Matrix Transpose

Transpose swaps rows and columns.

$$
A =
\begin{bmatrix}
1 & 2\\
3 & 4
\end{bmatrix}
$$

$$
A^T =
\begin{bmatrix}
1 & 3\\
2 & 4
\end{bmatrix}
$$

---

# 4. Task 2 — NumPy Implementation

**Duration:** 30 minutes

Create:

`linear_algebra.ipynb`

Implement the following operations WITHOUT using NumPy's built-in operation for the actual calculation.

You may use NumPy to create arrays and verify your results.

## Exercise 1: Vector Addition

Given:

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
```

Implement vector addition manually.

**Hint:** Add corresponding elements using a loop.

Expected result:

```text
[5, 7, 9]
```

## Exercise 2: Dot Product

Implement the dot product using a loop.

**Hint:**

$$
result = \sum_{i=1}^{n}a_ib_i
$$

Verify your implementation using `np.dot()`.

## Exercise 3: Matrix Multiplication

Given:

```python
A = np.array([
    [1, 2],
    [3, 4]
])

B = np.array([
    [5, 6],
    [7, 8]
])
```

Implement matrix multiplication manually.

**Hints:**

1. Select a row from A.
2. Select a column from B.
3. Calculate their dot product.
4. Store the result in the output matrix.

Do not use `np.matmul()` or the `@` operator in your implementation.

## Exercise 4: Matrix Transpose

Implement matrix transpose manually.

**Hint:** Rows become columns and columns become rows.

Verify your result using `.T`.

## Exercise 5: Shape Compatibility

Experiment with multiplying matrices of different shapes.

Answer:

1. Why can a $(2,3)$ matrix multiply a $(3,2)$ matrix?
2. Why can't a $(2,3)$ matrix multiply a $(2,3)$ matrix?
3. What determines the output dimensions?
4. What is the difference between matrix multiplication and element-wise multiplication?

---

# 5. Task 3 — Understanding Linear Regression

**Duration:** 30 minutes

Now that you understand vectors and matrices, let's build your first ML model.

## Problem Statement

You have data about students and their examination scores.

Your task is to predict examination scores based on study hours.

### Dataset

| Study Hours | Exam Score |
|---:|---:|
| 1 | 20 |
| 2 | 30 |
| 3 | 45 |
| 4 | 50 |
| 5 | 65 |
| 6 | 75 |
| 7 | 85 |

This is a supervised learning problem.

- Input: Study hours.
- Target: Examination score.
- Model: Linear Regression.

## 5.1 Understanding the Model

A linear regression model is represented as:

$$
\hat{y}=wx+b
$$

Where:

- $x$: Input feature.
- $\hat{y}$: Predicted output.
- $w$: Weight or slope.
- $b$: Bias or intercept.

The model needs to learn appropriate values of $w$ and $b$.

### Your task

Understand the effect of changing the parameters.

Try:

$$
\hat{y}=10x+5
$$

Then try:

$$
\hat{y}=5x+10
$$

Observe how changing the weight and bias changes the regression line.

## 5.2 Understanding the Loss Function

A prediction generally differs from the actual value.

We need a way to measure this difference.

Use Mean Squared Error (MSE):

$$
J(w,b)=\frac{1}{n}\sum_{i=1}^{n}
(\hat{y}_i-y_i)^2
$$

Where:

- $n$: Number of observations.
- $\hat{y}_i$: Predicted value.
- $y_i$: Actual value.

The objective is to minimize MSE.

### Understand

1. Why do we square the error?
2. Why do we calculate the average?
3. What happens when the prediction is perfect?
4. Why is MSE sensitive to large errors?

## 5.3 Understanding Gradient Descent

Gradient Descent is an optimization algorithm that updates model parameters to minimize the loss.

Weight update:

$$
w \leftarrow w-\alpha\frac{\partial J}{\partial w}
$$

Bias update:

$$
b \leftarrow b-\alpha\frac{\partial J}{\partial b}
$$

Where:

- $\alpha$: Learning rate.
- $J$: Loss function.
- $\partial J/\partial w$: Weight gradient.
- $\partial J/\partial b$: Bias gradient.

For Linear Regression with MSE:

$$
\frac{\partial J}{\partial w}
=
\frac{2}{n}\sum_{i=1}^{n}
(\hat{y}_i-y_i)x_i
$$

$$
\frac{\partial J}{\partial b}
=
\frac{2}{n}\sum_{i=1}^{n}
(\hat{y}_i-y_i)
$$

### Understand

- What is a gradient?
- Why do we need derivatives?
- Why do we subtract the gradient?
- What is the learning rate?
- What is convergence?

**Important:** Understand where these derivatives come from. Don't just memorize the formulas.

---

# 6. Task 4 — Implement Linear Regression From Scratch

**Duration:** 45 minutes

Create:

`linear_regression.py`

## Requirements

Implement the following yourself:

1. Dataset initialization.
2. Parameter initialization.
3. Forward propagation.
4. Loss calculation.
5. Gradient calculation.
6. Parameter updates.
7. Training loop.
8. Prediction function.

### Restrictions

You are allowed to use:

- Python
- NumPy
- Matplotlib

You are NOT allowed to use:

- Scikit-learn.
- TensorFlow.
- PyTorch.
- Existing Linear Regression implementations.

## Implementation Hints

### Hint 1: Initialize parameters

Start with:

```python
w = 0.0
b = 0.0

learning_rate = 0.01
epochs = 1000
```

### Hint 2: Forward propagation

Use:

$$
\hat{y}=wx+b
$$

Calculate predictions for every training observation.

### Hint 3: Calculate loss

Calculate the Mean Squared Error between predictions and actual values.

### Hint 4: Calculate gradients

Use the mathematical equations from Section 5.3.

Do not use numerical approximations for gradients. Implement analytical derivatives.

### Hint 5: Update parameters

Update both weight and bias using Gradient Descent.

### Hint 6: Training loop

Repeat the following for every epoch:

1. Forward propagation.
2. Calculate loss.
3. Calculate gradients.
4. Update parameters.
5. Store the loss.

### Hint 7: Prediction

Create a reusable function:

```python
def predict(x):
    # Return the model prediction
    pass
```

Use it to predict the score for a student who studies for 8 hours.

### Hint 8: Verify your implementation

Check that:

- Your parameters change during training.
- Your loss decreases.
- Your predictions are reasonable.
- Your model does not produce NaN or infinite values.

---

# 7. Task 5 — Experiments and Visualization

**Duration:** 15 minutes

Create two visualizations.

## Experiment A: Training Loss

Plot:

- X-axis: Epoch.
- Y-axis: MSE Loss.

Observe how the loss changes during training.

Questions:

1. Does the loss decrease?
2. Does it eventually stabilize?
3. What does convergence mean?
4. What happens if the loss increases continuously?

## Experiment B: Actual vs Predicted

Create a scatter plot of actual data points and overlay the regression line.

Observe whether the learned line captures the relationship between study hours and examination scores.

## Experiment C: Learning Rate

Run the model with different learning rates.

| Experiment | Learning Rate |
|---|---:|
| A | 0.0001 |
| B | 0.001 |
| C | 0.01 |
| D | 0.1 |

Record:

- Final loss.
- Number of epochs.
- Convergence behavior.
- Whether training diverges.

Do not assume that a higher learning rate always results in faster convergence.

---

# 8. Optional Challenge — Multiple Linear Regression

Extend your model to support multiple input features.

Instead of predicting examination scores using only study hours, use:

- Hours studied.
- Hours slept.
- Attendance percentage.

The model becomes:

$$
\hat{y}=Xw+b
$$

Where:

- $X$: Feature matrix.
- $w$: Weight vector.
- $b$: Bias.

For the MSE loss:

$$
J(w,b)=\frac{1}{n}\|Xw+b-y\|_2^2
$$

The gradients become:

$$
\nabla_w J=\frac{2}{n}X^T(Xw+b-y)
$$

$$
\frac{\partial J}{\partial b}
=\frac{2}{n}\sum_{i=1}^{n}(\hat{y}_i-y_i)
$$

### Hints

1. Represent the dataset as a matrix.
2. Represent weights as a vector.
3. Use matrix multiplication for predictions.
4. Derive the gradients with respect to the weight vector.
5. Update all weights simultaneously.

This will prepare you for understanding how neural networks process batches of data.

---

# 9. Project Structure

Your final Day 1 folder should look like this:

```text
day-01/
│
├── README.md
│
├── linear_algebra.ipynb
│
├── linear_regression.py
│
├── experiments/
│   ├── learning_rates.py
│   └── multiple_regression.py
│
├── outputs/
│   ├── loss_curve.png
│   └── predictions.png
│
└── notes/
    └── concepts.md
```

---

# 10. GitHub Deliverables

Your repository should contain:

- [ ] Linear algebra notebook.
- [ ] Manual vector operations.
- [ ] Manual matrix multiplication.
- [ ] Manual dot product.
- [ ] Linear Regression implementation.
- [ ] Analytical gradient calculations.
- [ ] Training loss visualization.
- [ ] Actual vs predicted visualization.
- [ ] Learning rate experiments.
- [ ] Mathematical explanations.
- [ ] Completed knowledge check.

Suggested repository name:

`ai-journey`

Suggested commit:

```bash
git add .
git commit -m "day 01: implement linear regression from scratch"
git push
```

---

# 11. Knowledge Check

Answer these questions in `notes/concepts.md`.

## Linear Algebra

1. What is the difference between a scalar, vector and matrix?
2. Explain the dot product geometrically and mathematically.
3. Why must matrix dimensions be compatible for multiplication?
4. What is the difference between matrix multiplication and element-wise multiplication?

## Machine Learning

5. What is supervised learning?
6. What does the weight represent in Linear Regression?
7. What is the role of bias?
8. Why do we need a loss function?
9. Why does MSE square the errors?

## Optimization

10. What is a gradient?
11. Explain Gradient Descent in your own words.
12. What happens when the learning rate is too high?
13. What happens when the learning rate is too low?
14. What does convergence mean?
15. Why do we update weights and bias separately?

## Implementation

16. Why do we store loss after every epoch?
17. Why should we separate training and prediction functions?
18. How would you extend the model to support multiple features?

---

# 12. Completion Criteria

Mark Day 1 complete only when:

- [ ] You can explain vectors and matrices without referring to notes.
- [ ] You can manually calculate a dot product.
- [ ] You understand matrix multiplication.
- [ ] You understand the Linear Regression equation.
- [ ] You can derive the gradients for MSE.
- [ ] You have implemented Gradient Descent yourself.
- [ ] Your model successfully trains.
- [ ] Your loss curve demonstrates learning.
- [ ] You have experimented with at least three learning rates.
- [ ] You have answered all knowledge-check questions.
- [ ] You have pushed your work to GitHub.

---

# 13. References and Learning Resources

## Mathematics

- 3Blue1Brown — Essence of Linear Algebra:
  https://www.3blue1brown.com/topics/linear-algebra

- Mathematics for Machine Learning:
  https://mml-book.github.io/

## NumPy

- NumPy Quickstart:
  https://numpy.org/doc/stable/user/quickstart.html

- NumPy Documentation:
  https://numpy.org/doc/

## Machine Learning

- Stanford CS229:
  https://cs229.stanford.edu/

- An Introduction to Statistical Learning:
  https://www.statlearning.com/

---

# 14. Expected Outcome

At the end of Day 1, you should have:

1. A working Linear Regression implementation.
2. A basic understanding of Linear Algebra.
3. An understanding of how Gradient Descent optimizes parameters.
4. Visual evidence of your model learning.
5. A documented GitHub contribution.

**Remember:** The objective is not simply to get correct predictions. It is to understand exactly how the model learns to produce them.

---

# Next: Day 02 — Probability, Statistics and Data Distributions

Topics:

- Probability fundamentals.
- Conditional probability.
- Bayes' theorem.
- Random variables.
- Probability distributions.
- Expected value and variance.
- Covariance and correlation.
- Statistical foundations for Machine Learning.
