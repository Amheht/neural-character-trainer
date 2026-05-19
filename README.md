# Neural Character Trainer

An interactive, browser-based neural network that learns to recognize hand-drawn characters. Draw on a 16×16 grid, train the network by labeling your drawings, and watch it learn to predict what you draw in real time. Every weight, connection, and activation is visible and editable.

No installation required. Open `index.html` in any modern browser.
Github Pages Link: **https://amheht.github.io/neural-character-trainer/**

---

## How to Use

### 1. Drawing

Use the **Draw Character** panel on the left to draw on the 16×16 grid.

- **Click and drag** to fill cells
- **Right-click and drag** to erase cells
- Toggle **✏ Draw / 🗑 Erase** to switch modes
- **Clear** wipes the grid
- **Invert** flips all filled and empty cells

---

### 2. Training

Type a character into the **Character** field in the Train panel (right side), then click **Add**.

This does two things simultaneously:
- Stores your drawing as a training example for that character
- Runs one training step, immediately updating the network weights

**Tips for good results:**
- Add at least 3–5 varied drawings per character — draw each one slightly differently to help the network generalize
- Characters that look visually similar (e.g. `I` and `l`, or `0` and `O`) will need more examples to distinguish
- The **Loss** value shown after each step tells you how wrong the network was — lower is better

#### Train 1×
Trains the current drawing on the selected character without storing it as a permanent example. Useful for fine-tuning.

#### Train All
Runs multiple full passes (epochs) over every stored example. Use this to significantly improve accuracy after collecting several examples. Set the epoch count higher (e.g. 500–1000) for better convergence.

---

### 3. Predicting

Once the network has at least one trained character, it predicts in real time as you draw. The **Live Prediction** panel shows the top 3 guesses with confidence percentages.

The confidence bars are computed using **softmax** — the network distributes 100% of its confidence across all known characters, so scores are always relative to each other.

---

### 4. Network Visualization

The middle panel shows the full network structure:

```
INPUT (16×16 grid)  →  HIDDEN LAYER (24 neurons)  →  OUTPUT LAYER (one node per character)
```

**Nodes** are colored by their current activation level — brighter means more strongly firing.

**Connection lines** between the hidden and output layers are colored by weight:
- **Green** — positive weight (this hidden neuron activates this output)
- **Red** — negative weight (this hidden neuron suppresses this output)
- **Thickness and opacity** indicate the weight's magnitude

By default, only connections with a weight above a small threshold are shown. Toggle **Show All Connections** to display every connection regardless of strength.

#### Selecting a Hidden Node

Click any hidden node to select it. This reveals:
- The node's current **activation value** and **bias**
- Its **weight heatmap** — a 16×16 visualization of how it weighs each input pixel. Green pixels excite the neuron; red pixels suppress it. This shows what visual feature the neuron has learned to detect
- Its connections to every output node, shown as clickable chips with their weight values

#### Selecting an Output Node

Click any output node to select it. This reveals:
- The node's **softmax confidence** and **bias**
- Its connections from every hidden node, with weights

#### Editing a Connection

Click directly on a **connection line** to open the connection editor. From here you can:
- **Edit the weight** — type any value and save
- **Randomize** — assign a new random weight
- **Delete** — removes the connection from both the visualization and the forward/backward pass. The line disappears and the connection no longer contributes to predictions or training. Click again on a deleted connection (shown as a faded chip) to restore it

---

### 5. Network Controls

Found in the right panel under **Network Controls**.

| Button | Effect |
|---|---|
| Randomize All Weights | Resets every weight in the network to small random values. Clears all learned features |
| Randomize Input → Hidden | Resets only the 6,144 weights between the input and hidden layer |
| Randomize Hidden → Output | Resets only the weights between hidden and output nodes |
| Reset Network | Wipes everything — weights, training data, and loss history — and starts fresh |

---

### 6. Save and Load

- **Save Session** — downloads a `.json` file containing the network weights, deleted connections, all stored training examples, and loss history
- **Load Session** — uploads a previously saved `.json` file and fully restores that state

Sessions are portable. You can save on one machine and load on another.

---

## How It Works

### Architecture

The network is a **feedforward neural network** with three layers:

| Layer | Size | Purpose |
|---|---|---|
| Input | 256 neurons | One per pixel. Value is 0 (empty) or 1 (filled) |
| Hidden | 24 neurons | Learns internal feature representations |
| Output | N neurons | One per trained character, added dynamically |

The total number of weights is `(256 × 24) + (24 × N)`, where N is the number of distinct characters trained.

---

### Forward Pass (Prediction)

**Input → Hidden**

Each hidden neuron computes a weighted sum of all 256 input pixels, adds a bias term, then passes the result through the **sigmoid function**:

```
σ(x) = 1 / (1 + e^(-x))
```

Sigmoid maps any number to a value strictly between 0 and 1, representing how strongly that neuron fired. The 24 hidden neurons each do this independently, producing 24 activation values.

**Hidden → Output**

Each output neuron computes a weighted sum of the 24 hidden activations, adds a bias, and passes through sigmoid again, producing a raw output value per character.

**Softmax (display only)**

The raw output values are then passed through **softmax** to produce the confidence percentages shown in the UI:

```
P(character i) = e^(raw_i) / Σ e^(raw_j)
```

Every character's exponentiated raw score is divided by the sum of all exponentiated scores. This forces all outputs to sum to exactly 1, creating a proper probability distribution. Softmax is used for display only — the actual learning uses the sigmoid values.

---

### Backward Pass (Learning)

When you provide a labeled example, the network runs **backpropagation**:

**1. Compute error**

The correct answer is represented as a one-hot vector — all zeros except a 1 at the correct character's position. The error for each output neuron is:

```
error = target - sigmoid_output
```

**2. Output deltas**

Each output neuron's error is scaled by the **sigmoid derivative**:

```
delta = error × sigmoid_output × (1 - sigmoid_output)
```

The sigmoid derivative is largest when a neuron is uncertain (output ≈ 0.5) and shrinks toward zero when it is already confident (output near 0 or 1). This prevents confident correct predictions from being aggressively updated.

**3. Update hidden → output weights**

```
weight += learning_rate × delta × hidden_activation
```

Weights connecting strongly-firing hidden neurons to wrong outputs are penalized most. Weights connected to inactive hidden neurons are barely touched — they didn't contribute to the error.

**4. Propagate error to hidden layer**

Hidden neurons have no direct target, so blame is distributed backward through the output weights:

```
hidden_error = Σ (output_delta × weight_to_that_output)
```

A hidden neuron that had a large weight to an output neuron that was very wrong receives a large error signal.

**5. Update input → hidden weights**

The same update rule applies, using the hidden delta and each pixel's value (0 or 1):

```
weight += learning_rate × hidden_delta × pixel_value
```

Because pixel values are binary, this simplifies to: only the weights for pixels that were *on* get updated. Dark pixels contribute nothing.

---

### What the Hidden Layer Learns

Each hidden neuron starts with random weights and no particular purpose. Through training, neurons whose weights happen to detect a useful visual feature — a vertical stroke, a curve, a closed loop — get reinforced, because those features are consistently useful for distinguishing characters. Neurons that detect nothing consistent drift toward zero and become effectively silent.

The **weight heatmap** (visible when a hidden node is selected) shows exactly what each neuron has learned to look for. A neuron that has learned to detect a horizontal crossbar will show bright green weights clustered in the middle rows of the grid.

---

### Learning Rate

The learning rate controls the step size of each weight update. It is adjustable via the slider in the Train panel.

- **Too high** — updates overshoot, weights oscillate, the network never converges
- **Too low** — learning is stable but very slow
- **Default (0.05)** — a reasonable starting point for this architecture

The **loss chart** tracks mean squared error over training steps. A healthy training run shows loss decreasing and flattening over time.
