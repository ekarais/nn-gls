# NN-GLS
A PyTorch implementation of [NN-GLS](https://arxiv.org/abs/2304.09157), a neural network estimation algorithm for geospatial data.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contents](#contents)
- [TODO](#TODO)
- [License](#license)


## Installation

Execute the following bash commands to install nn-gls.

```bash
# Clone the repository
git clone https://github.com/ekarais/nn-gls.git

# Navigate to the repository directory
cd nn-gls

# Create a Conda environment
conda create --name nn-gls --file conda_environment.yaml
conda activate nn-gls

# Install pip packages
pip install -r pip_packages.txt
```

## Usage
`tutorial.py` illustrates how to generate synthetic data using the Friedman function, fit an instance of NN-GLS and compute the learned kernel parameters as well as some performance metrics.

### Notes
- The authors give two choice for how to update the parameters of the kernel. I chose to synchronously update them alongside the MLP.
- I could not find details about the experiments with synthetic data. In particular, details such as the number of samples, considered values for kernel parameters, the architecture of the MLP, batch size, etc. seem to be omitted from the paper. Therefore, I had to assume some values for these.
- The train/test setup I implemented is not identical to what is described in the paper. I first generate a set of samples where the errors are sampled from the multivariate Gaussian with the covariance matrix implied by the Exponential kernel. I then split this set into train and test, and compute nearest neighbors within each set independently. 
- I observed a numerical stability issue when $\sigma_{init} \gt 0.8$. To remedy this, I set $\sigma_{init} = 0.1$, which also corresponds to the prior of low correlation between outcomes.
- I observed that at some point, the model starts overfitting on the proxy OLS loss: the MSE between decorrelated predictions and targets keeps going down while the MSE between the actual predictions and targets starts going up. This could perhaps be due to me forgetting to block some gradients.

## Contents
- `utilities.py`: Contains the implementation of NN-GLS as well as utility functions for creating synthetic data.
- `tutorial.ipynb`: A brief walkthrough showcasing data generation, model fitting and model evaluation.

## TODO
- [x] Create some plots for the training procedure.
- [ ] Improve documentation of `utilities.py`.
- [ ] Investigate if some gradients should be blocked.
- [ ] Write unit tests for components of NN-GLS.
- [ ] Enable training on GPU.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
