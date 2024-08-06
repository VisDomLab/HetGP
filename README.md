# HetGP
[CIKM 2024]

# 
## Overview

The codebase includes:
- **Neural Network Regression**: A simple feedforward neural network.
- **Gaussian Process Regression**: Both homoscedastic and heteroscedastic models.
- **Evidential Deep Learning**: For regression tasks with uncertainty quantification.

## Requirements

This project requires the following libraries:
- TensorFlow (`tensorflow`)
- NumPy (`numpy`)
- pandas (`pandas`)
- Scikit-learn (`scikit-learn`)
- TensorFlow Probability (`tensorflow_probability`)
- GPflow (`gpflow`)
- UCI Datasets Loader (`uci_datasets`)
- Evidential Deep Learning (`evidential_deep_learning`)
- tqdm (for progress bars in loops)

You can install requirements using:
```bash
pip install numpy pandas tensorflow scikit-learn tensorflow_probability gpflow tqdm evidential_deep_learning
```



To run the experiments with default settings:

```bash
python main.py
```


## Citation

If you find our work useful, please cite it as:

```bibtex
@inproceedings{udbhav,
Author = {Udbhav Mallanna,Dalavai and Dwivedi, Rajeev R and Thakur, Rini S and Kurmi, Vinod },
Title = {Quantifying Uncertainty
in Neural Networks through Residuals},
Booktitle = {CIKM},
Year   = {2024}
}
```

