# nncpp

A minimal neural network implementation in modern C++ designed primarily to demonstrate the mathematics behind **forward propagation** and **backpropagation**.

The current implementation provides:

* fully connected (`DenseLayer`) layers;
* convolution and max-pooling (`Conv2d`, `MaxPool2d`) layers;
* sigmoid and ReLU activations (`Sigmoid`, `ReLU`);
* summed half-squared-error loss (exposed through the `MSE` API);
* fused softmax and categorical cross-entropy for classification;
* gradient accumulation;
* mini-batch gradient descent;
* Xavier/Glorot weight initialization;
* a generic `Layer` interface for composing networks;
* a deterministic MNIST training example with test-set evaluation.

The implementation intentionally uses simple `std::vector<float>` tensors instead of a matrix library so that the code closely follows the underlying equations.

## Build and Test

The project requires a C++23 compiler and CMake 3.20 or newer.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure
./build/nncpp
```

The executable runs the 50-epoch convolutional MNIST example. The repository
includes the MNIST files under
`third_party/mnist/`; malformed or incomplete training and test splits are
rejected before training starts.

To run the same tests with AddressSanitizer enabled:

```bash
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug -DSANITIZER=address
cmake --build build-asan
ctest --test-dir build-asan --output-on-failure
```

## Network Model

The examples below are code fragments that assume `<array>`, `<cstddef>`, `<memory>`, `<ranges>`, and `nncpp/network.hpp` are included. A network can be constructed as:

```cpp
std::array<std::unique_ptr<nncpp::Layer>, 4> network = {
    std::make_unique<nncpp::DenseLayer>(nncpp::Shape{2}, nncpp::Shape{4}),
    std::make_unique<nncpp::Sigmoid>(4),
    std::make_unique<nncpp::DenseLayer>(nncpp::Shape{4}, nncpp::Shape{1}),
    std::make_unique<nncpp::Sigmoid>(1),
};
```

`Shape` records a tensor rank and dimensions. Spatial layers use conventional
`{channels, height, width}` (CHW) shapes; `total()` returns the number of scalar
values in the flattened `Tensor`. `CHWLayout` validates rank-3 shapes and provides
named dimension accessors plus checked `(channel, y, x)` indexing. Element-wise
layers preserve their input shape, while `DenseLayer` uses the complete input total.

This represents:

```text
x
│
▼
Dense(2 → 4)
│ z₁
▼
Sigmoid
│ a₁
▼
Dense(4 → 1)
│ z₂
▼
Sigmoid
│ a₂
▼
output
```

## Forward Propagation

### Dense Layer

For an input vector

$$
x=(x_1,\ldots,x_n)
$$

a dense layer computes:

$$
\boxed{z = Wx+b}
$$

For an individual output neuron $j$:

$$
z_j=b_j+\sum_i W_{ji}x_i
$$

This corresponds directly to `DenseLayer::forward()`:

```cpp
for (size_t outIndex = 0; outIndex < outputShape.total(); ++outIndex) {
    nncpp::Scalar z = biases[outIndex];
    
    for (size_t inIndex = 0; inIndex < inputShape.total(); ++inIndex) {
        const size_t wIndex = inIndex + inputShape.total() * outIndex;
        z += weights[wIndex] * input[inIndex];
    }
    
    output[outIndex] = z;
}
```

The weights are stored in a flat row-major array:

```text
W[output][input]

weights = {
    W00, W01, ...,
    W10, W11, ...,
    ...
}
```

with:

```cpp
wIndex = inIndex + inputShape.total() * outIndex;
```

The input is cached in `lastInput`, because it will later be required to calculate the weight gradients.

### Sigmoid Activation

A dense transformation alone is affine:

$$
z=Wx+b
$$

Multiple dense layers without activation functions can always be reduced to a single affine transformation. Nonlinear activation functions are therefore required to construct a nonlinear neural network.

The sigmoid activation is:

$$
\boxed{
\sigma(z)=\frac{1}{1+e^{-z}}
}
$$

and produces:

$$
a=\sigma(z)
$$

with:

$$
0<a<1
$$

`Sigmoid::forward()` performs this operation element-wise.

The complete forward pass for a two-layer network is therefore:

$$
z_1=W_1x+b_1
$$

$$
a_1=\sigma(z_1)
$$

$$
z_2=W_2a_1+b_2
$$

$$
a_2=\sigma(z_2)
$$

or:

$$
\boxed{
a_2=
\sigma\left(
W_2\sigma(W_1x+b_1)+b_2
\right)
}
$$

In code, the generic `Layer` interface allows the forward pass to simply iterate from left to right:

```cpp
nncpp::Tensor out = input;

for (const auto &layer : network) {
    out = layer->forward(out);
}
```

### ReLU Activation

ReLU is used by the MNIST hidden layer and is applied element-wise:

$$
\operatorname{ReLU}(z)=\max(0,z)
$$

Its backward pass forwards the incoming gradient where the cached activation
is positive and returns zero elsewhere. As with `Sigmoid`, call `forward()`
before each `backward()` call.

### Conv2d Layer

`Conv2d` applies learnable square kernels to a CHW tensor. Its constructor takes
the input shape, number of output channels, kernel size, stride, and padding:

```cpp
nncpp::Conv2d convolution(
    nncpp::Shape{1, 28, 28},
    8,
    3,
    1,
    nncpp::Conv2d::Padding::Same
);
```

For input height $H$, width $W$, kernel size $K$, stride $S$, and padding $P$
on each side, the spatial output dimensions are:

$$
H_{out}=\left\lfloor\frac{H+2P-K}{S}\right\rfloor+1,
\qquad
W_{out}=\left\lfloor\frac{W+2P-K}{S}\right\rfloor+1
$$

`Padding::Valid` uses no padding, `Padding::Same` preserves spatial dimensions
for an odd kernel with stride 1, and `Padding::Full` uses $K-1$ zeros on every
side. `backward()` returns the input gradient and accumulates kernel and bias
gradients. Call `resetGradients()` before a new batch and
`applyGradient(learningRate, batchSize)` after accumulating that batch.

### MaxPool2d Layer

`MaxPool2d` performs non-padded max pooling independently in every channel:

```cpp
nncpp::MaxPool2d pool(convolution.getOutputShape(), 2, 2);
```

Its output dimensions use the same formula with $P=0$. The forward pass caches
the selected input index for each pooling window. `backward()` routes the output
gradient to those indices; gradients are summed when overlapping windows select
the same input. Equal maxima deterministically select the first value visited.
The layer has no trainable parameters.

## Loss Functions

### Loss Interface

`Loss` is the polymorphic interface for scalar objectives:

```cpp
const nncpp::Scalar value = loss.forward(logits, target);
const nncpp::Tensor &gradient = loss.backward();
```

`forward()` evaluates one sample and caches the state needed by `backward()`,
which returns the gradient with respect to the loss input. Call `forward()`
before each `backward()` call. The standalone `MSE` functions predate this
interface and expose their derivative through `derivativeMSE()` instead.

### Summed Half-Squared Error

The current implementation uses summed half-squared error. The public functions retain the `MSE` name, but the result is not divided by the number of outputs and therefore is not a statistical mean.

For one output:

$$
E=\frac12(a-y)^2
$$

where $a$ is the network output and $y$ is the target.

Its derivative is:

$$
\boxed{
\frac{\partial E}{\partial a}=a-y
}
$$

For multiple outputs:

$$
E=\frac12\sum_i(a_i-y_i)^2
$$

The backward pass starts with:

```cpp
auto gradient = nncpp::derivativeMSE(out, targets);
```

At this point:

$$
\text{gradient} =
\frac{\partial E}{\partial a_L}
$$

where $a_L$ is the final network output.

### Softmax Categorical Cross-Entropy

For multi-class classification, `SoftmaxCategoricalCrossEntropy` accepts raw
logits and a target distribution whose values are nonnegative and sum to one.
The class count passed to its constructor must match both tensor sizes. Logits
and targets must be finite. It combines numerically stable softmax and
cross-entropy in one operation:

$$
E=-\sum_i y_i\log p_i,
\qquad
p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

The backward result is the gradient with respect to the logits. For a one-hot
target this simplifies to:

$$
\boxed{\frac{\partial E}{\partial z_i}=p_i-y_i}
$$

```cpp
nncpp::SoftmaxCategoricalCrossEntropy loss(10);
const nncpp::Scalar sampleLoss = loss.forward(logits, target);
const nncpp::Tensor &gradient = loss.backward();
```

Softmax subtracts the largest logit before exponentiation, avoiding overflow for
large finite inputs. `backward()` consumes the most recent forward result and
must not be called twice without another `forward()`.

## Backpropagation

Backpropagation repeatedly applies the chain rule while traversing the network in reverse order.

For a composition

$$
x\xrightarrow{f}y\xrightarrow{g}E
$$

the chain rule gives:

$$
\frac{\partial E}{\partial x} =
\frac{\partial E}{\partial y}
\frac{\partial y}{\partial x}
$$

Every `Layer::backward()` follows the same contract:

```text
input:   dE / d(layer output)
output:  dE / d(layer input)
```

Therefore, the entire backward pass can be expressed as:

```cpp
for (const auto& layer : std::views::reverse(network)) {
    gradient = layer->backward(gradient);
}
```

### Sigmoid Backward

For:

$$
a=\sigma(z)
$$

the sigmoid derivative is:

$$
\boxed{
\frac{\partial a}{\partial z} =
a(1-a)
}
$$

If the next operation supplies:

$$
\frac{\partial E}{\partial a}
$$

then the chain rule gives:

$$
\boxed{
\frac{\partial E}{\partial z} =
\frac{\partial E}{\partial a}
a(1-a)
}
$$

This is implemented directly by `Sigmoid::backward()`:

```cpp
inputGradient[i] = outputGradient[i] * output[i] * (1.0f - output[i]);
```

The sigmoid output from the forward pass is cached because it is required during backward propagation.

### Dense Layer Backward

For a dense layer:

$$
y=Wx+b
$$

assume that backward propagation provides:

$$
g=
\frac{\partial E}{\partial y}
$$

The layer must calculate three gradients.

#### Weight Gradient

For an individual weight:

$$
y_j=\sum_iW_{ji}x_i+b_j
$$

therefore:

$$
\frac{\partial y_j}{\partial W_{ji}}=x_i
$$

and:

$$
\boxed{
\frac{\partial E}{\partial W_{ji}} =
\frac{\partial E}{\partial y_j}x_i
}
$$

In the implementation:

```cpp
weightGradients[wIndex] += lastInput[inIndex] * grad;
```

#### Bias Gradient

Because:

$$
\frac{\partial y_j}{\partial b_j}=1
$$

we obtain:

$$
\boxed{
\frac{\partial E}{\partial b_j} =
\frac{\partial E}{\partial y_j}
}
$$

Implemented as:

```cpp
biasGradients[outIndex] += grad;
```

#### Input Gradient

The gradient must also continue toward the preceding layer.

Since:

$$
\frac{\partial y_j}{\partial x_i}=W_{ji}
$$

we obtain:

$$
\boxed{
\frac{\partial E}{\partial x_i} =
\sum_j
W_{ji}
\frac{\partial E}{\partial y_j}
}
$$

or, in matrix notation:

$$
\boxed{
\frac{\partial E}{\partial x} =
W^T
\frac{\partial E}{\partial y}
}
$$

Implemented as:

```cpp
inputGradient[inIndex] += weights[wIndex] * grad;
```

`DenseLayer::backward()` returns this `inputGradient`, allowing backpropagation to continue through the preceding layer.

## Complete Backward Pass

For:

```text
Dense1 → Sigmoid1 → Dense2 → Sigmoid2
```

forward propagation is:

```text
x
│
▼
Dense1
│ z₁
▼
Sigmoid1
│ a₁
▼
Dense2
│ z₂
▼
Sigmoid2
│ a₂
▼
E
```

Backpropagation follows exactly the opposite direction:

```text
dE/da₂
   │
   ▼
Sigmoid2.backward()
   │
   │ dE/dz₂
   ▼
Dense2.backward()
   │
   │ dE/da₁
   ▼
Sigmoid1.backward()
   │
   │ dE/dz₁
   ▼
Dense1.backward()
   │
   ▼
dE/dx
```

Thus the implementation directly represents the chain rule:

```cpp
auto gradient = nncpp::derivativeMSE(out, targets);

for (const auto& layer : std::views::reverse(network)) {
    gradient = layer->backward(gradient);
}
```

## Gradient Descent

`DenseLayer::backward()` accumulates parameter gradients rather than immediately modifying parameters.

After backpropagation, gradient descent updates the parameters:

$$
\boxed{
W \leftarrow W-\eta\frac{\partial E}{\partial W}
}
$$

$$
\boxed{
b \leftarrow b-\eta\frac{\partial E}{\partial b}
}
$$

where $\eta$ is the learning rate.

This corresponds to:

```cpp
layer->applyGradient(learningRate, batchSize);
```

Gradients are explicitly cleared before accumulating a new set:

```cpp
layer->resetGradients();
```

`Layer` provides no-op implementations of `applyGradient()` and
`resetGradients()`. This lets the generic loop call both operations on every
layer; `DenseLayer` and `Conv2d` override them, while activation and pooling
layers require no parameter update.

The separation between

```text
forward
backward
applyGradient
resetGradients
```

allows gradients to be accumulated across multiple samples, which can later be used for mini-batch training.

## Training Loop

The complete single-sample training procedure is:

```cpp
const nncpp::Tensor input = {0.25f, 0.5f};
const nncpp::Tensor targets = {1.0f};
constexpr nncpp::Scalar learningRate = 0.01f;
constexpr nncpp::Scalar errorEpsilon = 1e-4f;

for (std::size_t iteration = 0; iteration < 1'000'000; ++iteration) {
    nncpp::Tensor out = input;

    // Forward propagation
    for (const auto& layer : network) {
        out = layer->forward(out);
    }

    if (nncpp::loss(out, targets) < errorEpsilon) {
        break;
    }

    // Clear old gradients
    for (const auto& layer : network) {
        layer->resetGradients();
    }

    // dE / d(network output)
    auto gradient =
        nncpp::derivativeMSE(out, targets);

    // Backpropagation
    for (const auto& layer : std::views::reverse(network)) {
        gradient = layer->backward(gradient);
    }

    // Gradient descent
    for (const auto& layer : network) {
        layer->applyGradient(learningRate, 1);
    }
}
```

Conceptually:

```text
input
  │
  │ forward
  ▼
prediction
  │
  │ loss
  ▼
error
  │
  │ derivative
  ▼
output gradient
  │
  │ backward
  ▼
parameter gradients
  │
  │ gradient descent
  ▼
updated parameters
```

## MNIST Training Example

`MNISTTrainingConv2dExample()` builds a
`1×28×28 → Conv2d(8, 3×3) → ReLU → MaxPool2d(2×2) → Dense(10)` classifier with
fused softmax cross-entropy loss. Each epoch shuffles training indices with a
reproducible `std::mt19937` sequence seeded with `42`. Input and one-hot target
buffers are reused, while trainable-layer gradients accumulate over 32 samples
and are scaled to the batch mean before the parameter update.

After every epoch, the example reports average online training loss and
fixed-test-set loss and accuracy:

```text
epoch: 0, train loss: 0.41234, test loss: 0.21567, test accuracy: 93.45000%
```

Evaluation performs forward passes only and does not update parameters or
accumulated gradients.

## Weight Initialization

Dense layer weights use Xavier/Glorot uniform initialization:

$$
W\sim
U\left(
-\sqrt{\frac{6}{n_{in}+n_{out}}},
+\sqrt{\frac{6}{n_{in}+n_{out}}}
\right)
$$

This avoids initializing every neuron with identical weights and is well suited to sigmoid-based networks.

Biases are initially zero.

## Summary

For every layer, backpropagation follows one simple rule:

$$
\boxed{
\text{receive gradient of output}
\rightarrow
\text{apply local derivative}
\rightarrow
\text{return gradient of input}
}
$$

For `Sigmoid`:

$$
\boxed{
\frac{\partial E}{\partial z} =
\frac{\partial E}{\partial a}a(1-a)
}
$$

For `DenseLayer`:

$$
\boxed{
\frac{\partial E}{\partial W} =
\frac{\partial E}{\partial y}x^T
}
$$

$$
\boxed{
\frac{\partial E}{\partial b} =
\frac{\partial E}{\partial y}
}
$$

$$
\boxed{
\frac{\partial E}{\partial x} =
W^T\frac{\partial E}{\partial y}
}
$$

The network-level backward pass is therefore simply the repeated application of these local derivatives in reverse layer order.
