Fashion-MNIST (Xiao et al. 2017) and MNIST (LeCun et al.) idx files are downloaded by scripts/prepare_data.py from
the zalandoresearch/fashion-mnist and fgnt/mnist GitHub repositories (byte-identical to the canonical files) and
converted to npz.  Breast Cancer Wisconsin (Diagnostic) is loaded from scikit-learn.  Official train/test splits are
preserved: training sub-samples come from the training split only; the entire official test split of each class pair
is used for testing.
