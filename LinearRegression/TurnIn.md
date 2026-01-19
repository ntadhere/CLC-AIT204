# Linear Regression and Differentiation Essentials

## TOC

- [Math Segment](#1-interpret--geometrically)
- [The Code](#the-code)
    - [Repository](https://github.com/ntadhere/CLC-AIT204)
    - [Streamlit Site]()
- [Ethical Considerations](#ethical-considerations)

### 1. Interpret $\frac{dy}{dx}$ geometrically
- This is the slope of the tangent line to the curve $y=f(x)$ at a point $x=a$ 
- It represents the instantaneous rate of change of $y$ with respect to $x$
- Positive slope: curve increasing
- Negative slope: curve decresing
- Shope 0: horizontal tangent (often local max/min)
- Formally, it is the limit of secant slopes:
$$
\left.\frac{dy}{dx}\right|_{x=a}
=
\lim_{h \to 0}
\frac{f(a+h) - f(a)}{h}
$$

**Source**: base on in class note and using of chatgpt paraphrasing

---

### 2. How many differentiation formulas do we have and what are they?

1. Constant Rule: 

$$
\frac{d}{dx}(c) = 0 
$$

2. Power Rule:

$$
\frac{d}{dx}(x^n) = n x^{\,n-1} \quad \text{(real } n \text{, where defined)} 
$$

3. Constant Multiple Rule: 

$$ 
\frac{d}{dx}(cf) = c f' 
$$

4. Sum/Differrence Rule: 

$$
\frac{d}{dx}(f \pm g) = f' \pm g'
$$

5. Product Rule: 

$$
(fg)' = f'g + fg'
$$

6. Quotient Rule: 

$$
\frac{d}{dx}\bigl(f(g(x))\bigr) = f'(g(x))\,g'(x)
$$

7. Chain Rule: 

$$
\frac{d}{dx}\bigl(f(g(x))\bigr) = f'(g(x))\,g'(x)
$$

8.  Exponential Rule:

$$
\frac{d}{dx}(e^x) = e^x
$$

$$
\frac{d}{dx}(a^x) = a^x \ln a
$$

9. Logarithmic Rule:

$$
\frac{d}{dx}(\ln x) = \frac{1}{x}
$$

$$
\frac{d}{dx}(\log_a x) = \frac{1}{x \ln a}
$$

10. Trigonometric Fuctions:

$$
(\sin x)' = \cos x
$$

$$
(\cos x)' = -\sin x
$$

$$
(\tan x)' = \sec^2 x
$$

$$
(\csc x)' = -\csc x \cot x
$$

$$
(\sec x)' = \sec x \tan x
$$

$$
(\cot x)' = -\csc^2 x
$$

**source**: Padlet "AIT-204 Derivatives"

---


### 3. Differentiatie:

$$
y = 4 + 2x - 3x^2 - 5x^3 - 8x^4 + 9x^5
$$
Differentiatie term-by-term:
- $(4)' = 0$
- $(2x)' = 2$
- $(-3x^2)' = -6x$
- $(-5x^3)' = -15x^2$
- $(-8x^4)' = -32x^3$
- $(9x^5)' = 45x^4$

So,
$$
y' = 2 - 6x - 15x^2 - 32x^3 + 45x^4
$$
---
---
$$
y = \frac{1}{x} + \frac{3}{x^2} + \frac{2}{x^3}
= x^{-1} + 3x^{-2} + 2x^{-3}
$$
Use the power rule 
$$
y' = (-1)x^{-2} + 3(-2)x^{-3} + 2(-3)x^{-4}
$$

So,
$$
y' = -x^{-2} - 6x^{-3} - 6x^{-4}
$$
$$
y' = -\frac{1}{x^2} - \frac{6}{x^3} - \frac{6}{x^4}
$$
---
---
$$
y = \sqrt[3]{3x^2} - \frac{1}{\sqrt{5x}}
$$
$$
y = (3x^2)^{1/3} - (5x)^{-1/2}
$$

First Term:

$$
\frac{d}{dx}(u^{1/3}) = \frac{1}{3}u^{-2/3}u',
\quad u = 3x^2,\quad u' = 6x
$$
$$
\Rightarrow \frac{d}{dx}(3x^2)^{1/3}
= \frac{1}{3}(3x^2)^{-2/3}(6x)
= 2x(3x^2)^{-2/3}
$$

Second Term:
$$
\frac{d}{dx}\bigl(-(5x)^{-1/2}\bigr)
= -\left[(-\tfrac{1}{2})(5x)^{-3/2}\cdot 5\right]
= \frac{5}{2}(5x)^{-3/2}
$$

Final Answer:
$$
y' = 2x(3x^2)^{-2/3} + \frac{5}{2}(5x)^{-3/2}
$$
---

### 4. Define partial derivative:
- A partial derivative measure how a multivariable fuction changes with respect to one variable while holding the other variables constant.
- If $z=f(x,y)$ then:
$$
\frac{\partial z}{\partial x} \text{ treats } y \text{ as a constant,}
\quad
\frac{\partial z}{\partial y} \text{ treats } x \text{ as a constant.}
$$

---
### 5. Given the following fuctions find $\frac{\partial z}{\partial x}$ and $\frac{\partial z}{\partial y}$

$$
z = 2x^2 - 3xy + 4y^2
$$
$$
\frac{\partial z}{\partial x} = 4x - 3y,
\qquad
\frac{\partial z}{\partial y} = -3x + 8y
$$
---
$$
z = \frac{x^2}{y} + \frac{y^2}{x}
$$

- With respect to y 
$$
z = x^2 y^{-1} + y^2 x^{-1}
$$
$$
\frac{\partial z}{\partial x}
= \frac{\partial}{\partial x}(x^2 y^{-1})
+ \frac{\partial}{\partial x}(y^2 x^{-1})
= 2x y^{-1} + y^2(-1)x^{-2}
$$

- With respect to y:
$$
\frac{\partial z}{\partial x}
= \frac{2x}{y} - \frac{y^2}{x^2}
$$
$$
\frac{\partial z}{\partial y}
= \frac{\partial}{\partial y}(x^2 y^{-1})
+ \frac{\partial}{\partial y}(y^2 x^{-1})
= x^2(-1)y^{-2} + 2y x^{-1}
$$
$$
\frac{\partial z}{\partial y}
= -\frac{x^2}{y^2} + \frac{2y}{x}
$$
---
$$
z = e^{x^2 + xy}
$$
Let:
$$
u = x^2 + xy, \quad z = e^u
$$
Then
$$
\frac{\partial z}{\partial x} = e^u \frac{\partial u}{\partial x},
\qquad
\frac{\partial z}{\partial y} = e^u \frac{\partial u}{\partial y}
$$
$$
\frac{\partial u}{\partial x} = 2x + y,
\qquad
\frac{\partial u}{\partial y} = x
$$
So,
$$
\frac{\partial z}{\partial x} = e^{x^2 + xy}(2x + y),
\qquad
\frac{\partial z}{\partial y} = x e^{x^2 + xy}
$$

---

## The Code

We made a [Streamlit Site]() that allows the user to view a linear regression model based on a synthetic data set. The user is able to change the learning rate and the number of iterations and then see how those changes affect the model.

### [Code Repository](https://github.com/ntadhere/CLC-AIT204)

### [Streamlit Site]()

## Ethical Considerations

### 1. Bias in Data Generation and Model Fairness

Bias can be introduced at multiple stages of data generation and collection. If the dataset underrepresents certain groups or reflects historical inequalities, the model may systematically perform worse for those populations. Synthetic or simulated data can also embed designer assumptions that unintentionally favor particular outcomes. This can lead to unfair predictions, disparate error rates, or reinforcement of existing social biases. Addressing this requires examining data sources, understanding who or what is excluded, and evaluating model performance across relevant subgroups rather than relying solely on aggregate metrics.

### 2. Data Privacy and Use of Real-World Data

When real-world data is used, especially data involving individuals, there is a risk of violating privacy through improper collection, storage, or use. Even anonymized datasets can sometimes be re-identified when combined with other information. Ethical practice includes minimizing the amount of personal data collected, removing direct identifiers, securing data storage, and ensuring compliance with relevant legal and institutional guidelines. Consent and clarity about how data will be used are also central considerations.

### 3. Model Transparency and Explainability

Models that operate as “black boxes” can make it difficult to understand why certain predictions are produced. This lack of transparency can undermine trust and make it harder to identify errors, biases, or inappropriate behavior. Explainability is especially important when model outputs influence decisions that affect people. Using interpretable models when possible, or applying explanation techniques to more complex models, helps stakeholders assess whether the model’s behavior aligns with domain knowledge and ethical expectations.

### 4. Responsible Use and Societal Impact

Models should be applied in ways that provide clear benefits while minimizing potential harm. This includes considering the context in which predictions are used and avoiding deployment in high-stakes settings without sufficient validation. Ethical responsibility also involves anticipating misuse, such as over-reliance on automated predictions or application outside the model’s intended scope. Framing model outputs as decision-support tools rather than definitive judgments helps maintain appropriate human oversight.

### 5. Mitigating Harm from Prediction Errors

No model is perfectly accurate, and errors can have real consequences depending on the application. False positives and false negatives may affect different groups in different ways. Ethical model design involves understanding which types of errors are most harmful, setting thresholds accordingly, and clearly communicating uncertainty. Regular monitoring, retraining, and fallback mechanisms can reduce the long-term impact of incorrect predictions.