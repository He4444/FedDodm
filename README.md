# FedDodm
Code for the article Personalized Federated Learning with Enhanced Implicit Generalization

Abstract—Integrating personalization into federated learning is crucial for addressing data heterogeneity and surpassing the limitations of a single aggregated model. Personalized federated learning excels at capturing inter-client similarities and meeting diverse client needs through custom-made models. However, even with personalized approaches, it’s essential to aggregate knowledge among clients to ensure universal benefits. This paper proposes Federated Dual Objectives and Dual Models (FedDodm), a novel approach that employs two independent models to separately address explicit personalization and implicit generalization objectives in personalized federated learning. By treating these objectives as distinct loss functions and training models accordingly, we achieve a balance between the two through a fusion method. Extensive experiments across various models and learning tasks demonstrate that FedDodm outperforms state-of-the-art federated learning approaches, marking a significant advancement in effectively integrating personalized and generalized knowledge.

## Usage:
```python
cd ./system
python main.py -data MNIST -m CNN -algo FedDomo -gr 2000 -did 0  # using the MNIST dataset, the FedDomo
```

## Generate Dataset：
If you need to generate the corresponding dataset, run the following commands:
```python
python generate_MNIST.py iid - -  # for IID and unbalanced scenario
python generate_MNIST.py iid balance -  # for IID and balanced scenario
python generate_MNIST.py noniid - pat  # for pathological Non-IID and unbalanced scenario
```
