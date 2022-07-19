
About PAPER : 
# PAPER
> ### Neural Collaborative Filtering

### Description
* dataset/ : Dataset
* pretrained/ : Save Pretrained GMF, MLP models to initialize NCF model
* (GMF || MLP)_data_.py  : Load data, Create Dataset for (GMF || MLP) model
* (GMF || MLP || NCF)_model_.py : (GMF || MLP || NCF) Model
* (GMF || MLP)_main_.py : Main code to make (GMF || MLP) model and Save (GMF || MLP) model in directory pretrained/
* data_.py : Load data, Create Dataset for NCF model
* main.py : Main code for NCF model - Load pretrained GMF, MLP model and optimize NCF model

### How to run this python 
```
python GMF_main_.py
python MLP_main_.py
python main.py
```

### reference
* https://arxiv.org/pdf/1708.05031.pdf
