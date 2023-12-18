# Fed-QSSL


## Requirments
```
pip install -r requirements.txt
```

## Main Training Command
1. Federated Quantized SSL experiment with our proposed scheme ```python3 src/decentralized_ssl_main_he.py --dataset=cifarssl --epochs=500 --gpu=1 --iid=0 --dirichlet --dir_beta=0.1```
2. Local bitwidth configuration can be specified by modifying ```cbit_local``` list

## File Structure
```angular2html
├── ...
├── Dec-SSL
|   |── data 			# training data
|   |── src 			# source code
|   |   |── options 	# parameters and config
|   |   |── sampling 	# different sampling regimes for non-IIDness
|   |   |── update 	    # pipeline for each local client
|   |   |── models 	    # network architecture
|   |   |── *_main 	    # main training and testing scripts
|   |   └── ...
|   |── save 			# logged results
|   |── scripts 	# experiment scripts
|   |── misc 			# related scripts for finetuning 
└── ...
```

## Acknowledgements:
1. [FL](https://github.com/AshwinRJ/decentralized-Learning-PyTorch)
2. SSL ([1](https://github.com/SsnL/moco_align_uniform), [2](https://github.com/leftthomas/SimCLR), [3](https://github.com/PatrickHua/SimSiam), [4](https://github.com/HobbitLong/PyContrast), [5](https://github.com/IcarusWizard/MAE), [6](https://github.com/liruiw/Dec-SSL))
3. Quantized training ([7](https://github.com/lucamocerino/Binary-Neural-Networks-PyTorch-1.0))
