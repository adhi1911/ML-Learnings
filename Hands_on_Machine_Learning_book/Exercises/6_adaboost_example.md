## Example Question

Consider a small dataset with 4 instances and 2 features. We have 2 weak classifiers (predictors) and their predictions for each instance. The initial weights for each instance are equal.

### Dataset
| Instance | Feature 1 | Feature 2 | True Label |
|----------|-----------|-----------|------------|
| 1        | 1         | 2         | 1          |
| 2        | 2         | 1         | 1          |
| 3        | 1         | 1         | 0          |
| 4        | 2         | 2         | 0          |

### Weak Classifier Predictions
| Instance | Predictor 1 | Predictor 2 |
|----------|-------------|-------------|
| 1        | 1           | 1           |
| 2        | 0           | 1           |
| 3        | 0           | 0           |
| 4        | 1           | 0           |

## Steps to Calculate AdaBoost Predictions

1. **Initialize weights**: \( w(i) = \frac{1}{m} \) for all \( i \), where \( m \) is the number of instances.
2. **Calculate weighted error rate** for each predictor using Equation 7-1.
3. **Calculate predictor weight** using Equation 7-2.
4. **Update weights** using Equation 7-3.
5. **Calculate final prediction** using Equation 7-4.

## Solution

1. **Initialize weights**:
   \[
   w(1) = w(2) = w(3) = w(4) = \frac{1}{4} = 0.25
   \]

2. **Calculate weighted error rate** for each predictor:
   - For Predictor 1:
     \[
     r_1 = \sum_{i=1}^{4} w(i) \cdot \mathbb{1}( \hat{y}_1(i) \neq y(i) ) = 0.25 \cdot 0 + 0.25 \cdot 1 + 0.25 \cdot 0 + 0.25 \cdot 1 = 0.5
     \]
   - For Predictor 2:
     \[
     r_2 = \sum_{i=1}^{4} w(i) \cdot \mathbb{1}( \hat{y}_2(i) \neq y(i) ) = 0.25 \cdot 0 + 0.25 \cdot 0 + 0.25 \cdot 0 + 0.25 \cdot 1 = 0.25
     \]

3. **Calculate predictor weight**:
   - For Predictor 1:
     \[
     \alpha_1 = \frac{1}{2} \log \left( \frac{1 - r_1}{r_1} \right) = \frac{1}{2} \log \left( \frac{1 - 0.5}{0.5} \right) = \frac{1}{2} \log(1) = 0
     \]
   - For Predictor 2:
     \[
     \alpha_2 = \frac{1}{2} \log \left( \frac{1 - r_2}{r_2} \right) = \frac{1}{2} \log \left( \frac{1 - 0.25}{0.25} \right) = \frac{1}{2} \log(3) \approx 0.55
     \]

4. **Update weights**:
   - For Predictor 1 (since \( \alpha_1 = 0 \), it has no effect):
     \[
     w(i) \leftarrow w(i) \quad \text{for all } i
     \]
   - For Predictor 2:
     \[
     w(i) \leftarrow \begin{cases} 
     w(i) & \text{if } \hat{y}_2(i) = y(i) \\
     w(i) \exp(\alpha_2) & \text{if } \hat{y}_2(i) \neq y(i)
     \end{cases}
     \]
     - For instance 4, where \( \hat{y}_2(4) \neq y(4) \):
       \[
       w(4) \leftarrow w(4) \exp(\alpha_2) = 0.25 \exp(0.55) \approx 0.25 \cdot 1.73 = 0.4325
       \]
     - Normalize weights:
       \[
       w(1) = w(2) = w(3) = \frac{0.25}{1.4325} \approx 0.1745, \quad w(4) = \frac{0.4325}{1.4325} \approx 0.3025
       \]

5. **Calculate final prediction**:
   \[
   \hat{y}(x) = \arg\max_k \sum_{j=1}^{2} \alpha_j \cdot \mathbb{1}( \hat{y}_j(x) = k )
   \]
   Since \( \alpha_1 = 0 \) and \( \alpha_2 \approx 0.55 \), the final prediction is dominated by Predictor 2:
   \[
   \hat{y}(x) = \hat{y}_2(x)
   \]

### Final Predictions
Using Predictor 2's predictions:
| Instance | Final Prediction |
|----------|------------------|
| 1        | 1                |
| 2        | 1                |
| 3        | 0                |
| 4        | 0                |

Thus, the final predictions for the instances are [1, 1, 0, 0].