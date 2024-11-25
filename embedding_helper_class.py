"""
Module: Embedding Generators

This module contains classes and functions for generating text and image embeddings using 
pre-trained models (BERT for text and VGG16 for images). It also provides methods for 
normalizing embeddings and processing image files in bulk.

Classes:
    TextEmbeddingGenerator: Generates vector embeddings for text using BERT.
    ImageEmbeddingGenerator: Generates vector embeddings for images using VGG16.

Functions:
    normalize_vector: Applies normalization (L2 or Min-Max) to a vector.
    get_clean_text_from_url: Fetches and cleans text from a given URL.
    get_website_text_as_embedding: Fetches website text from URL and converts it into an embedding.
    text_to_vector: Converts a given text into a vector embedding.
    get_image_vector_embedding: Converts an image into a vector embedding.
    images_to_vector_json_file: Converts images in a folder to vector embeddings and saves them in a JSON file.
"""

from transformers import BertTokenizer, BertModel
import os
import json
import torch
import re
import torchvision.transforms as transforms
from torchvision.models import vgg16, VGG16_Weights
from PIL import Image
from transformers import logging
import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass, field
import numpy as np

logging.set_verbosity_error()


@dataclass
class TextEmbeddingGenerator:
    """
    Class for generating text embeddings using a pre-trained BERT model.

    Attributes:
        model_name (str): The name of the pre-trained BERT model.
        normalize (bool): Indicates whether to normalize the embeddings.
        normalization_method (str): The method of normalization to use ('L2' or 'min-max').
        tokenizer (BertTokenizer): Tokenizer for the BERT model (initialized after object creation).
        model (BertModel): Pre-trained BERT model (initialized after object creation).
    """
    model_name: str = 'bert-base-uncased'
    normalize: bool = False
    normalization_method: str = "L2"
    tokenizer: BertTokenizer = field(init=False)
    model: BertModel = field(init=False)

    def __post_init__(self):
        """Initializes the BERT tokenizer and model."""
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertModel.from_pretrained(self.model_name, ignore_mismatched_sizes=True)
        self.model.eval()

    def normalize_vector(self, vector):
        """
        Applies normalization (L2 or Min-Max) to the provided vector.

        Args:
            vector (list): The vector to be normalized.

        Returns:
            list: The normalized vector.

        Raises:
            ValueError: If an unsupported normalization method is specified.
        """
        if self.normalization_method == "L2":
            norm = np.linalg.norm(vector)
            if norm == 0:
                return vector
            return [val / norm for val in vector]

        elif self.normalization_method == "min-max":
            min_val = min(vector)
            max_val = max(vector)
            if max_val - min_val == 0:
                return vector
            return [(val - min_val) / (max_val - min_val) for val in vector]

        else:
            raise ValueError(f"Unsupported normalization method: {self.normalization_method}")

    def get_clean_text_from_url(self, url):
        """
        Fetches and cleans text from a given URL.

        Args:
            url (str): The URL to fetch text from.

        Returns:
            str: The cleaned text.
        """
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text()
        return text.replace("\n", " ").replace("\r", " ").strip()

    def get_website_text_as_embedding(self, url, print_to_console=False):
        """
        Fetches text from a URL and converts it into an embedding.

        Args:
            url (str): The URL to fetch text from.
            print_to_console (bool): Whether to print the text to the console.

        Returns:
            list: The generated embedding for the website text.
        """
        clean_text = self.get_clean_text_from_url(url)

        if print_to_console:
            print("\n###")
            print("URL: " + url)
            print(clean_text)

        return self.text_to_vector(clean_text)

    def text_to_vector(self, text, normalize=False):
        """
        Converts a given text into a vector embedding using the BERT model.

        Args:
            text (str): The text to convert.
            normalize (bool): Whether to normalize the resulting embedding.

        Returns:
            list: The generated text embedding.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        text = text.lower().strip()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

        def chunk_text(text, max_tokens=512):
            words = text.split()
            chunks = [' '.join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]
            return chunks

        text_chunks = chunk_text(text)
        chunk_embeddings = []

        for chunk in text_chunks:
            inputs = self.tokenizer(chunk, return_tensors='pt', max_length=512, truncation=True).to(device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            chunk_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
            chunk_embeddings.append(chunk_embedding)

        final_embedding = np.mean(chunk_embeddings, axis=0)

        if self.normalize:
            return self.normalize_vector(final_embedding)
        else:
            return final_embedding


@dataclass
class ImageEmbeddingGenerator:
    """
    Class for generating image embeddings using a pre-trained VGG16 model.

    Attributes:
        normalize (bool): Indicates whether to normalize the embeddings.
        normalization_method (str): The method of normalization to use ('L2' or 'min-max').
        base_model (torch.nn.Module): Pre-trained VGG16 model (initialized after object creation).
    """
    normalize: bool = False
    normalization_method: str = "L2"
    base_model: torch.nn.Module = field(init=False)

    def __post_init__(self):
        """Initializes the pre-trained VGG16 model for image embedding generation."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        weights = VGG16_Weights.IMAGENET1K_V1
        self.base_model = vgg16(weights=weights).to(device)
        self.base_model.eval()
        self.base_model.classifier = torch.nn.Identity()

    def normalize_vector(self, vector):
        """
        Applies normalization (L2 or Min-Max) to the provided vector.

        Args:
            vector (list): The vector to be normalized.

        Returns:
            list: The normalized vector.

        Raises:
            ValueError: If an unsupported normalization method is specified.
        """
        if self.normalization_method == "L2":
            norm = np.linalg.norm(vector)
            if norm == 0:
                return vector
            return [val / norm for val in vector]

        elif self.normalization_method == "min-max":
            min_val = min(vector)
            max_val = max(vector)
            if max_val - min_val == 0:
                return vector
            return [(val - min_val) / (max_val - min_val) for val in vector]

        else:
            raise ValueError(f"Unsupported normalization method: {self.normalization_method}")

    def get_image_vector_embedding(self, filepath):
        """
        Generates an image vector embedding from a given image file.

        Args:
            filepath (str): The path to the image file.

        Returns:
            list: The generated image embedding.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        img = Image.open(filepath).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            vector_embedding = self.base_model(img_tensor).cpu().numpy().flatten()

        if self.normalize:
            return self.normalize_vector(vector_embedding)
        return vector_embedding

    def images_to_vector_json_file(self, images_folder='db_images', targetFile='image_vector_embeddings.json'):
        """
        Converts all images in a folder to vector embeddings and saves them in a JSON file.

        Args:
            images_folder (str): The folder containing the images.
            targetFile (str): The file to save the image embeddings as JSON.

        Raises:
            Exception: If there's an error while processing images or saving to the JSON file.
        """
        embeddings_list = []
        for filename in os.listdir(images_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(images_folder, filename)
                try:
                    vector_embedding = self.get_image_vector_embedding(img_path)
                    # Convert NumPy array or float32 elements to Python native types
                    vector_embedding_list = [float(val) for val in vector_embedding]
                    embeddings_list.append({"filename": filename, "vector_embedding": vector_embedding_list})
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    continue

        try:
            with open(targetFile, 'w') as f:
                json.dump(embeddings_list, f, indent=4)
            print(f"All vector embeddings saved to {targetFile}")
        except Exception as e:
            print(f"Error saving to {targetFile}: {e}")
