# modsel
A simple model selection tool for machine learning experiments.

DISCLAIMER: This is a general purpose tool, but several choices are currently tailored to my personal use-cases. 

WARNING: the API is still subject to changes.

## The idea

The guiding principle of this tool is a complete decoupling between your experiments and the hyperparameters search. In three steps:

 1. You design the code of your experiments by assuming to already have the best hyperparameters (no search involved)
 2. You accept a given command line argument to provide hyperparameters, and you print the validation loss on stdout (no dependency on this library)
 3. You define a yaml hyperparameter grid and you run `modsel` over your script
 
If your experiment can accept a hyperparametrization and can emit a scalar measure of how good it is, then you can use this tool.
 
## Minimal example
 
Your code `experiment.py` (make sure it is executable):
 
```python
#!/usr/bin/env python3
import argparse
import ruamel.yaml as yaml
from sklearn.datasets import load_boston
from sklearn.linear_model import Ridge

# Read the hyperparameters. You can make this argument optional
# and fall back to some default values if not provided.
argparser = argparse.ArgumentParser()
argparser.add_argument('--hp', type=str)
args = argparser.parse_args()
hp = yaml.safe_load(args.hp)

# Load a dataset and train a model
X, y = load_boston(return_X_y=True)
clf = Ridge(alpha=hp['alpha'], normalize=hp['normalize'])
clf.fit(X[:250], y[:250])

# Test the classifier and adjust the sign of the loss
loss = float(-clf.score(X[250:], y[250:]))

# Log the loss
print("---")
print(yaml.dump({'Validation loss': loss}))
print("---")
```

It is important to log the loss as a YAML string enclosed by `---`.
You can log whatever you want: the only mandatory field is `Validation loss`,
a scalar which indicates the performance of the hyperparametrization. Lower
losses correspond to better models.

The hyperparameters grid (`grid.yaml`):
 
```yaml
normalize:
  - True
  - False
alpha: [0.1, 1.0, 10.0, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
```
 
Run with `modsel grid.yaml experiment.py --searches 30`. The software will
select the best hyperparameters and will print them to stdout:

```
...

Hyperparameters:
{'alpha': 1000.0, 'normalize': False}

Measurements:
[{'Validation loss': -0.590815454310589}]
```

## Installing

To install the unstable, development version:

    pip install git+https://github.com/danieleds/modsel.git
    
You will then find `modsel` in your `$PATH`.
