# marketplace_products

---
title: Marketplace Products
emoji: 🚀
colorFrom: purple
colorTo: yellow
sdk: gradio
sdk_version: 4.37.2
app_file: app.py
pinned: false
license: apache-2.0
---

Check out the configuration reference at https://huggingface.co/spaces/Gilvan/marketplace_products/tree/main

# This model is a fine-tuned version of distilbert/distilbert-base-uncased for NER Task. 

### The following hyperparameters were used during training:

- learning_rate=2e-5,
- per_device_train_batch_size=128,
- per_device_eval_batch_size=128,
- num_train_epochs=4,
- weight_decay=0.01,
- num_epochs: 4

### It achieves the following results on the evaluation set:

#### Training results

| Epoch | Training Loss | Validation Loss | Precision |  Recall  |    F1    | Accuracy |
|:-----:|:-------------:|:---------------:|:---------:|:--------:|:--------:|:--------:|
|   1   |    0.286800   |     0.125389    |  0.977472 | 0.977472 | 0.977472 | 0.977472 |
|   2   |    0.051800   |     0.026039    |  0.993742 | 0.993742 | 0.993742 | 0.993742 |
|   3   |    0.024200   |     0.013263    |  0.994994 | 0.994994 | 0.994994 | 0.994994 |
|   4   |    0.015500   |     0.013252    |  0.994994 | 0.994994 | 0.994994 | 0.994994 |


# HOWTO

- Clone the repository by: git clone https://github.com/gilvanveras/marketplace_products.git
- Install the requirements.txt by: pip install -r requirements.txt
- Download the files model.safetensors and optimizer.pt inside the folder 'ner_model' from this URL: https://huggingface.co/spaces/Gilvan/marketplace_products/tree/main/ner_model
- Execute the file app.py by: python app.py

Your application is running :)

![image](https://github.com/gilvanveras/marketplace_products/assets/15756603/53c3e268-0c2e-4d9f-9f23-f3dd70f8308b)


Framework versions
Transformers 4.41.2
Pytorch 2.3.0+cu121
Datasets 2.19.2
Tokenizers 0.19.1
