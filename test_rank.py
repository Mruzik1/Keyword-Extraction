import re
import os
import sys
import time
import json
from tqdm import tqdm
import shutil
import pandas as pd

import kagglehub
import networkx as nx
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# download nltk dependencies
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')


def download_dataset(path2download):
    """ Download dataset """
    path = kagglehub.dataset_download("snehaanbhawal/resume-dataset")
    for f in os.listdir(path):
        try:
            shutil.move(f"{path}\\{f}", path2download)
        except shutil.Error as e:
            print(f"WARNING: {e}")
    shutil.rmtree(path)
    print("Path to dataset files:", path2download)
    return path2download


def get_text_csv(csv_path):
    """ Extracts and combines a text from csv (category-wise) """
    df = pd.read_csv(csv_path)
    texts = dict()
    for c in df["Category"].unique():
        text = df[df["Category"] == c]["Resume_str"]
        texts[c] = "\n".join(text)
    return texts


def preprocess_text(text):
    """ Text preprocessing """
    text = text.lower()
    # remove punctuation and digits
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # tokenization
    tokens = nltk.word_tokenize(text, language='english')
    # remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens


def build_graph(tokens, window_size=4):
    """ Creates a graph for neighboring tokens """
    graph = nx.Graph()

    for idx, word in enumerate(tokens):
        for j in range(idx + 1, idx + window_size):
            if j < len(tokens):
                neighbor = tokens[j]
                if word != neighbor:
                    graph.add_edge(word, neighbor)
    return graph


def pagerank_algorithm(graph, damping=0.85, max_iter=100, tol=1.0e-6):
    """ Pagerank implementation """
    nodes = list(graph.nodes())
    N = len(nodes)
    pagerank = dict.fromkeys(nodes, 1.0 / N)
    damping_value = (1.0 - damping) / N
    end_iter = 0

    for iteration in range(max_iter):
        new_pagerank = dict.fromkeys(nodes, damping_value)
        for node in nodes:
            for neighbor in graph.neighbors(node):
                out_degree = len(list(graph.neighbors(neighbor)))
                if out_degree > 0:
                    new_pagerank[node] += damping * (pagerank[neighbor] / out_degree)
        # check for convergence
        diff = sum(abs(new_pagerank[node] - pagerank[node]) for node in nodes)
        if diff < tol:
            break
        pagerank = new_pagerank
        end_iter = iteration
    return pagerank, end_iter


def print_keywords(keywords):
    """ Prints a list of keywords with ranks """
    for i in keywords:
        print(f"\t- {i[0]} [{round(i[1], ndigits=4)}]")


def extract_keywords(text_dict, top_n=10):
    """ Uses a dict with texts separated by categories to extract keywords for each category """
    top_keywords = dict()
    for c in tqdm(text_dict, total=len(text_dict)):
        tokens = preprocess_text(text_dict[c])
        graph = build_graph(tokens)
        pagerank, end_iter = pagerank_algorithm(graph)
        print(f" End iteration for {c} - {end_iter}")
        keywords = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
        top_keywords[c] = keywords[:top_n]
        print(f"Top keyword for {c}:")
        print_keywords(top_keywords[c])
    return top_keywords


def clean_keywords(keywords):
    """ Remove common keywords for each category """
    keywords_sets = [{i[0] for i in keywords[k]} for k in keywords]
    common_keywords = list(set.intersection(*keywords_sets))
    for k in keywords:
        keywords[k] = list(filter(lambda x: x[0] not in common_keywords, keywords[k]))
    return keywords


if __name__ == "__main__":
    start_time = time.time()

    if len(sys.argv) > 1:
        data_directory = sys.argv[1]
    else:
        data_directory = download_dataset("./data")

    texts = get_text_csv(f"{data_directory}/Resume/Resume.csv")
    keywords = extract_keywords(texts, top_n=15)
    keywords_clean = clean_keywords(keywords)

    with open("result.json", "w") as f:
        json.dump(keywords_clean, f)

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Execution Time: {execution_time_minutes}m")