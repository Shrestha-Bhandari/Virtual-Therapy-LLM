import torch
from langchain_community.llms import CTransformers
from langchain import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from textblob import TextBlob
import numpy as np
import pandas as pd

# FAISS Path (ensure this matches the path used in conversation.py)
DB_FAISS_PATH = r"C:\Masters\NLP Project-20241206T163105Z-001\NLP Project\Data\vectorstores\db_faiss"

class ModelEvaluator:
    def __init__(self, models_to_evaluate):
        """
        Initialize evaluator with multiple models and retrieval setup.
        """
        self.models = {}
        self.embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2', 
            model_kwargs={'device': 'cpu'}
        )
        
        self.db = FAISS.load_local(DB_FAISS_PATH, self.embeddings, allow_dangerous_deserialization=True)
        
        for name, model_info in models_to_evaluate.items():
            self.models[name] = CTransformers(
                model=model_info['path'], 
                model_type=model_info.get('model_type', 'llama'),
                max_new_tokens=model_info.get('max_new_tokens', 512),
                temperature=model_info.get('temperature', 1.0),
                top_k=model_info.get('top_k', 50),
                top_p=model_info.get('top_p', 0.95)
            )

    def generate_responses(self, texts):
        """
        Generate responses from each model for the given input texts.
        """
        responses = {name: [] for name in self.models.keys()}
        for model_name, model in self.models.items():
            for text in texts:
                # Retrieve relevant documents
                retriever = self.db.as_retriever(search_kwargs={'k': 1})  # Adjust as needed
                relevant_docs = retriever.get_relevant_documents(text)
                
                # Prepare context for the response
                context = "\n".join([doc.page_content for doc in relevant_docs]) if relevant_docs else ""
                
                # Combine context with the user input and pass to the model
                response = model(f"Context: {context}\nUser's Current Message: {text}")
                responses[model_name].append(response)
                
        return responses

    def calculate_perplexity(self, texts, responses):
        """
        Calculate perplexity for the generated responses.
        """
        results = {}
        for model_name, model_responses in responses.items():
            perplexities = []
            try:
                for prompt, response in zip(texts, model_responses):
                    perplexities.append(len(response.split()) / len(prompt.split()))  # Proxy calculation
                results[model_name] = sum(perplexities) / len(perplexities)
            except Exception as e:
                print(f"Perplexity calculation error for {model_name}: {e}")
                results[model_name] = None
        return results

    def sentiment_consistency(self, responses):
        """
        Evaluate sentiment consistency for the generated responses.
        """
        results = {}
        sentiments = []
        for response in responses:
            if not response.strip():  # Handle empty responses
                response = "No meaningful response generated."
            blob = TextBlob(response)
            sentiments.append(blob.sentiment.polarity)
        # Calculate consistency metrics
        results['mean_sentiment'] = np.mean(sentiments)
        results['sentiment_std'] = np.std(sentiments)
        results['sentiment_variance'] = np.var(sentiments)
        return results

    def evaluate(self, texts):
        """
        Comprehensive model evaluation.
        """
        # Generate responses for each model
        generated_responses = self.generate_responses(texts)
        
        # Evaluate perplexity and sentiment
        perplexity_scores = self.calculate_perplexity(texts, generated_responses)
        
        results = {
            'Model': [],
            'Perplexity': [],
            'Mean Sentiment': [],
            'Sentiment Std Dev': [],
            'Sentiment Variance': []
        }

        for model_name, model_responses in generated_responses.items():
            sentiment_results = self.sentiment_consistency(model_responses)
            results['Model'].append(model_name)
            results['Perplexity'].append(perplexity_scores[model_name])
            results['Mean Sentiment'].append(sentiment_results['mean_sentiment'])
            results['Sentiment Std Dev'].append(sentiment_results['sentiment_std'])
            results['Sentiment Variance'].append(sentiment_results['sentiment_variance'])

        return pd.DataFrame(results)

def main():
    # Define models to evaluate
    models_to_evaluate = {
        'Llama-2-7B': {
            'path': "TheBloke/Llama-2-7B-Chat-GGML",
            'model_type': 'llama',
            'max_new_tokens': 512,
            'temperature': 0.9
        },
        'Mistral-7B': {
            'path': "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
            'model_type': "mistral",
            'max_new_tokens': 512,
            'temperature': 0.7,
            'top_k': 50,
            'top_p': 0.95
        }
    }

    # Evaluation texts (actual chatbot prompts)
    evaluation_texts = [
        "How does personal recovery differ from clinical recovery?",
        "What does it mean to have a life beyond a mental health diagnosis?",
        "Can you explain the three definitions of recovery mentioned by Ruth Ralph and Patrick Corrigan?"
        "What are some strategies for maintaining hope during mental health treatment?"
        "People don’t understand what I’m going through.",
         "How do I plan for my recovery?",
         "Does spirituality help in recovery?",
         "Can relationships help in recovery?",
         "How can I feel better day to day?",
         "What does recovery mean in mental health?"

    ]

    # Initialize evaluator
    evaluator = ModelEvaluator(models_to_evaluate)

    # Run evaluation
    results = evaluator.evaluate(evaluation_texts)

    # Display results
    print("Model Evaluation Results:")
    print(results)

    # Optional: Save results to CSV
    results.to_csv('model_evaluation_results.csv', index=False)

if __name__ == '__main__':
    main()
