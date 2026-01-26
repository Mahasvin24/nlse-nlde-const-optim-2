# Temporal Arithmetic for Energy-Efficient Convolutions: nLSE/nLDE Constant Optimization

This repository implements and extends the temporal arithmetic approach presented in the ASPLOS 2024 paper **"Energy Efficient Convolutions with Temporal Arithmetic"** by Gretsch et al. [Paper Link](https://sites.cs.ucsb.edu/~sherwood/pubs/ASPLOS-24-temparith.pdf)

## 📖 Background

### Temporal Arithmetic & Delay Space

Traditional digital computation operates in "importance space," where values are represented as voltages or digital numbers. The ASPLOS paper introduces a revolutionary **delay space** representation where:

- Values are encoded as **timing of signal edges** rather than voltage levels
- A negative log transformation converts importance space to delay space: `delay = -log(value)`
- **Multiplication becomes addition** in delay space (implemented as simple delay)
- **Addition becomes nLSE** (negative Log-Sum-Exponential) in delay space

This encoding achieves **>2× energy efficiency** and **4 orders of magnitude improvement** in energy-delay product for near-sensor convolution operations.

### Key Operations

#### nLSE (negative Log-Sum-Exponential)
Performs addition in delay space:
```
nLSE(x', y') = -log(exp(-x') + exp(-y'))
```
Where `x'` and `y'` are delay-space values. This is approximated using min/max operations and tunable constants C and D:
```
nLSE(x', y') ≈ min(x', y', max(x'+C₁, y'+D₁), ..., max(x'+Cₙ, y'+Dₙ))
```

#### nLDE (negative Log-Difference-Exponential)
Performs subtraction in delay space using the inhibit operation:
```
nLDE(x', y') = -log(exp(-x') - exp(-y'))
```
Approximated similarly with constants E and F.

## 🎯 Project Goals

This project extends the original paper by:

1. **Reimplementing** the temporal arithmetic framework in PyTorch
2. **Optimizing constants** (C, D, E, F) for improved accuracy
3. **Noise robustness**: Learning constants that perform well under realistic noise conditions
4. **Evaluation**: Comprehensive benchmarking of accuracy vs. energy trade-offs

## 📁 Project Structure

```
nlse-nlde-const-optim-2/
├── constants/                    # Learned and original constants
│   ├── orig_constants.pt        # Original constants from paper
│   ├── learned_constants.pt     # Optimized constants
│   ├── load_orig_constants.py   # Load paper constants
│   └── view_constants.py        # Visualize constants
├── optimizers/                   # Training and optimization
│   └── nlse_optimizer.py        # Adam-based constant optimizer
├── tests/                        # Validation and testing
│   ├── nlse.py                  # nLSE accuracy tests
│   ├── nlde.py                  # nLDE accuracy tests
│   └── bucket_distribution.py   # Testing bucket dist gen
├── graphs/                       # Visualization tools
│   ├── data_distribution.py     # Data distribution plots
│   └── fig_4_temporal_addition.py # Reproduce paper figures
├── utils/                        # Core utilities
│   ├── temporal_artithmetic.py  # nLSE/nLDE implementations
│   ├── helpers.py               # Utility functions
│   └── distributions.py         # Distribution generator
└── requirements.txt             # Dependencies
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- PyTorch
- NumPy
- Matplotlib

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd nlse-nlde-const-optim-2

# Install dependencies
pip install -r requirements.txt
```

> **Important**: All commands should be run from the project root directory (`nlse-nlde-const-optim-2/`) since the project uses `utils` as a package.

## 💻 Usage

> **Note**: Since the project uses `utils` as a package, all scripts must be run as modules from the project root using the `-m` flag.

### Testing Original Constants

Test the accuracy of the original paper constants:

```bash
python3 -m tests.nlse
```

This evaluates nLSE accuracy across different numbers of max_terms (1-10, 15, 20) and generates an accuracy plot.

### Optimizing Constants

Train improved constants using gradient descent:

```bash
python3 -m optimizers.nlse_optimizer
```

**Hyperparameters** (in `nlse_optimizer.py`):
- `num_epochs`: Training iterations (default: 1000)
- `batch_size`: Samples per epoch (default: 100,000)
- `learning_rate`: Adam optimizer LR (default: 0.01)
- `max_terms`: Number of approximation terms (1-20)

The optimizer:
1. Generates uniform random samples in [0, 1]
2. Converts to delay space: `x' = -log(x)`
3. Computes exact addition: `-log(exp(-x') + exp(-y'))`
4. Compares with nLSE approximation
5. Minimizes RNRMSE (Range-Normalized Root Mean Squared Error)
6. Saves improved constants to `constants/learned_constants.pt`

### Noise Robustness Training

Enable noisy training in `nlse_optimizer.py`:

```python
trainer = nLSETrainer(
    model=model,
    noisy=True,  # Enable noise injection
    ...
)
```

Noise is injected:
- **Pre-VTC**: Gaussian noise on importance space values
- **Post-VTC**: Gaussian noise on delay space values

### Visualization

View and compare constants:

```bash
python3 -m constants.view_constants
```

Generate paper figures:

```bash
python3 -m graphs.fig_4_temporal_addition
```

## 📊 Results

The optimizer tracks:
- **Error before training**: Using original constants
- **Error after training**: Using learned constants
- **Improvement**: Percentage reduction in error

Example output:
```
Error for  5 maxterms: 1.23450% -> 0.98765% (improvement 0.24685%)
Error for 10 maxterms: 0.45678% -> 0.39012% (improvement 0.06666%)
```

Typical accuracy (with optimal max_terms):
- **max_terms=10**: ~99.5% accuracy
- **max_terms=20**: >99.8% accuracy

## 🔬 Key Implementation Details

### nLSE Implementation

From `utils/temporal_artithmetic.py`:

```python
def nlse(x_p: torch.Tensor, y_p: torch.Tensor, C: torch.Tensor, D: torch.Tensor):
    # K-shift to avoid negative delays
    K = -torch.min(torch.cat((C, D)))
    x_p = x_p + K
    y_p = y_p + K
    
    # Ensure x_p >= y_p for consistency
    x_p, y_p = torch.maximum(x_p, y_p), torch.minimum(x_p, y_p)
    
    # Compute approximation terms
    X = x_p + C  # shape: (N, max_terms)
    Y = y_p + D  # shape: (N, max_terms)
    maximum_terms = torch.maximum(X, Y)
    
    # Final nLSE: minimum over all terms
    all_terms = torch.cat((x_p, y_p, maximum_terms), dim=1)
    nlse_result = torch.min(all_terms, dim=1)[0]
    
    return nlse_result - K
```

### Error Metric

Uses **RNRMSE** (Range-Normalized RMSE) as in the paper:

```python
def rnrmse(pred, target):
    rmse = torch.sqrt(torch.mean((pred - target)**2))
    data_range = target.max() - target.min()
    return rmse / data_range
```

## 📈 Reproducing Paper Results

To reproduce Figure 4 (Temporal Addition Accuracy):

```bash
python3 -m graphs.fig_4_temporal_addition
```

To analyze error distributions:

```bash
python3 -m tests.bucket_distribution
```

## 🛠️ Advanced Features

### Custom Constants

Edit `constants/load_orig_constants.py` to define custom C and D values:

```python
C = {
    5: torch.tensor([[-0.91, -1.30, -1.82, -2.58, -3.89]]),
}
D = {
    5: torch.tensor([[-0.62, -0.40, -0.24, -0.12, -0.04]]),
}
```

Then regenerate the constants file:

```bash
python3 -m constants.load_orig_constants
```

### GPU Acceleration

The code automatically detects and uses available accelerators:
- CUDA (NVIDIA GPUs)
- MPS (Apple Silicon)
- CPU (fallback)

To force CPU usage:
```python
device = torch.device('cpu')
```

## 📚 Citation

If you use this code or build upon this work, please cite the original paper:

```bibtex
@inproceedings{gretsch2024temporal,
  title={Energy Efficient Convolutions with Temporal Arithmetic},
  author={Gretsch, Rhys and Song, Peiyang and Madhavan, Advait and Lau, Jeremy and Sherwood, Timothy},
  booktitle={29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (ASPLOS '24)},
  year={2024},
  month={April},
  location={La Jolla, CA, USA},
  publisher={ACM},
  doi={10.1145/3620665.3640395}
}
```

## 🔍 Key Insights from the Paper

1. **Energy Efficiency**: Temporal encoding reduces energy per convolution by >2× compared to state-of-the-art digital approaches

2. **Simplicity**: Complex multiplication becomes simple delay; addition uses only min/max gates

3. **Sensor Integration**: Natural fit with staged ADC readout in modern cameras (replace ADC with voltage-to-time converters)

4. **Recurrence Architecture**: Operations can be chained spatially or recurrently in time

5. **Noise Tolerance**: The approximation is robust to delay element noise and quantization

## 🤝 Contributing

Contributions are welcome! Areas for exploration:
- Adaptive constant selection based on input distributions
- Hardware-aware constant optimization
- Extension to 2D convolution operations
- Integration with actual VTC hardware models

## 📝 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

- Original paper authors: Rhys Gretsch, Peiyang Song, Advait Madhavan, Jeremy Lau, and Timothy Sherwood
- UC Santa Barbara Computer Science Department
- University of Maryland

## 📧 Contact

For questions or collaborations, please open an issue on the repository.

---

**Note**: This is a research implementation. For production use, hardware-specific optimizations and thorough validation are recommended.
