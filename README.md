# Generative Pseudo Labeling (GPL)


Generative pseudo labelling is a semi-supervised machine learning technique that involves using both labelled and unlabelled data to improve model performance. The goal of the technique is to use the unlabelled data to generate additional labelled data that can be used to train the model. It can be used to perform domain adaptation of a previously trained model.

At a high level, GPL consists of three data preparation steps and one fine-tuning step. We will begin by looking at the data preparation portion of GPL. The three steps are:

- **Query Generation:** Generating queries for passages using a pretrained model.
- **Negative Mining;** Retrieveing similar passages to the generated queriies that do not match.
- **Pseudo Labelling:** Using a pretrained cross encoder model to ensure the quality of the queries and passages pairs generated in the previous steps.

Once the data is prepared any biencoder model of choice can be finetuned using it.

The benefits of using generative pseudo labelling are that it allows us to make use of large amounts of unlabelled data, which can be cheaper and faster to obtain than labelled data. Additionally, by incorporating the unlabelled data, the model can learn more generalizable features and reduce overfitting to the labelled data. 
