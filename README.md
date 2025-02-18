# Fine-Tuning of Gemma-2B Using KerasNLP with LoRA

## General

### AIM
The aim of this project is to fine-tune the GEMMA model on a general Q&A dataset from Hugging Face to improve its ability to provide accurate and contextually relevant answers to a variety of questions.

### Motivation
The main goal of this project is to learn more about Natural Language Processing (NLP), as I’m really interested in the field. I want to dive into fine-tuning language models and understand how it works, as it’s a key part of improving models for different tasks. This is a personal project for now, and looking forward in future to work on projects together with team. For now, this is just a basic step to learn and grow in the field.

### Introduction to Google's Gemma
Gemma is a family of lightweight, state-of-the art open models from Google, built from the same research and technology used to create the Gemini models. They are text-to-text, decoder-only large language models, available in English, with open weights, pre-trained variants, and instruction-tuned variants. Gemma models are well-suited for a variety of text generation tasks, including question answering, summarization, and reasoning. Their relatively small size makes it possible to deploy them in environments with limited resources such as a laptop, desktop or your own cloud infrastructure, democratizing access to state of the art AI models and helping foster innovation for everyone.

### But why only Gemma ?
I chose Gemma 2B because it performs really well in NLP benchmarks, often beating other popular open-source models like LLaMA and Mistral. Its advanced architecture, open-source availability, and the fact that it provides easy-to-follow code make it a great fit for hands-on learning, especially for someone like me who is getting started with fine-tuning.

## Project Specifics

### Dataset & Preprocessing
The dataset used for fine-tuning the Gemma-2B model is the Databricks Dolly 15K dataset, available from [Hugging Face](https://huggingface.co/datasets/databricks/databricks-dolly-15k). This dataset consists of instructional and response pairs for training language models.
Preprocessing:
- Context Filtering: To simplify the dataset, we are ignoring  'context', as context is not required for this particular fine-tuning task.
- Sequence Length Limitation: The input sequence length was capped at 256 tokens to manage memory usage efficiently.
- Data Subset: Only 1000 examples from the dataset were used for quicker experimentation and testing.

### Model Configuration & Training Setup

The model used in this project is Gemma-2B as mentioned before.

Customizations:
- LoRA (Low-Rank Adaptation): LoRA was enabled with a rank of 4 to reduce memory usage and allow for more efficient training by modifying only a small subset of model parameters.
- Sequence Length Limitation: The input sequence was limited to 256 tokens to control memory usage during training.


Training Setup:
- Optimizer: The AdamW optimizer was used, with a learning rate of 5e-5 and a weight decay of 0.01 to optimize model performance.
- Loss Function: SparseCategoricalCrossentropy was chosen for multi-class classification, where the model predicts the next token in a sequence.
- Metrics: SparseCategoricalAccuracy was used to evaluate the accuracy of the model’s predictions during training.
- Training Parameters:
    - Epochs: 1 epoch (for fast experimentation)
    - Batch Size: 1 (for memory efficiency)

### Training & Evaluation

Training the Model: The model was trained on the dataset with the SparseCategoricalCrossentropy loss function and the AdamW optimizer. The training process focuses on minimizing the loss and adjusting the model's weights to make more accurate next-token predictions.

Evaluation Approach: Instead of using a traditional validation or test set, the model's performance was evaluated by generating answers from input prompts during inference. This qualitative evaluation helps assess how well the model generates coherent, relevant, and contextually appropriate responses.

- Inference Evaluation: After training, the model was tested by providing prompts and observing the quality of the generated responses. The focus here was on the coherence, relevance, and naturalness of the text produced, rather than relying on numerical accuracy metrics like token prediction.

This method allows us to gauge how well the model "understands" and generates meaningful content, which is crucial for tasks such as dialogue systems or language generation.

### Results & Future Work

Results:
The model was evaluated based on the quality of the generated responses, focusing on relevance, coherence, and contextual accuracy instead of traditional accuracy metrics.

- Good Responses: Coherent, contextually relevant answers suggest the model is learning effectively.
- Areas for Improvement: Nonsensical or irrelevant responses indicate room for improvement.

Insights:

- The model could perform better with a larger dataset (beyond 1000 examples) and more diverse data to improve its ability to handle a broader range of queries. (The input data feeded was limited to make training faster as the finetuning was done on Google Colab's normal version)

## Conclusion:

This is a basic version of the model, and there’s definitely room for improvement. Moving forward, I plan to enhance the quality of the model, fine-tune it with more diverse and larger datasets, and explore deeper techniques in NLP. The journey of continuous learning and improvement in this field is exciting, and I look forward to refining the model to achieve even better results.

#### Jay Shri Krishna!
