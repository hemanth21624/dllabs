# Deep Learning Lab Programs

10 Deep Learning Lab programs organized in simple, exam-friendly Python files.

## Programs

1. XOR using Multilayer Perceptron (MLP)
2. Regularization using parameter penalties, augmentation and noise
3. Comparison of SGD, Momentum and Adam
4. CNN on MNIST
5. LSTM-based text classification
6. RNN vs Bidirectional RNN
7. Encoder-Decoder LSTM for sequence-to-sequence learning
8. Restricted Boltzmann Machine (RBM)
9. Denoising Autoencoder
10. GAN on MNIST

## Notes

- Programs 1, 3, 4, 5, 6 and 9 use a consistent `Sequential()` + `model.add()` style where practical.
- Program 7 uses the Functional API because the decoder must receive the encoder states.
- Program 8 implements the RBM manually using contrastive divergence.
- Program 10 uses a custom training loop with `GradientTape`, which is appropriate for a GAN.
- Exact numerical outputs can vary because neural networks use random initialization and random data/noise.
