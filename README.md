## Notes
- paper gives a choice for how to update the parameters of the kernel, i chose to synchronously update it as it seemed simpler
- i couldnt find details about the experiments, in particular things like number of samples, considered values for kernel parameters, the architecture of the MLP, batch size, etc. so i made these up.
- i observe a numerical stability issue when sigma is initialized with >~0.8, I deliberately start it from 0.1, which sort of corresponds to the prior of low/no correlation between units.
- train/test setup is not identical to paper
- i observe that at some point, the model starts overfitting on the proxy OLS loss -> the MSE between decorrelated predictions and targets keeps going down while the MSE between the actual predictions and targets starts going up. this could perhaps be because i am forgetting to block some gradients.

## Todos
- investigate if some gradients should be blocked
- write unit tests for components of NN-GLS
- make trainable on GPU
- create some plots for the training procedure