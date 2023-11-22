# NN-GLS
A PyTorch implementation of [NN-GLS](https://arxiv.org/abs/2304.09157), a neural network estimation algorithm for geospatial data.

## Notes
- The authors give two choice for how to update the parameters of the kernel. I chose to synchronously update them alongside the MLP.
- I could not find details about the experiments with synthetic data. In particular, details such as the number of samples, considered values for kernel parameters, the architecture of the MLP, batch size, etc. seem to be omitted from the paper. Therefore, I had to assume some values for these.
- The train/test setup I implemented is not identical to what is described in the paper. For testing, I do not make use of the training set at all. 
- I observed a numerical stability issue when $\sigma_{init} \gt 0.8$. To remedy this, I set $\sigma_{init} = 0.1$, which also corresponds to the prior of low correlation between outcomes.
- I observed that at some point, the model starts overfitting on the proxy OLS loss: the MSE between decorrelated predictions and targets keeps going down while the MSE between the actual predictions and targets starts going up. This could perhaps be due to me forgetting to block some gradients.

## Todos
- [ ] Investigate if some gradients should be blocked.
- [ ] Write unit tests for components of NN-GLS.
- [ ] Enable training on GPU.
- [ ] create some plots for the training procedure