import os
import requests
from tqdm import tqdm
import pandas as pd
import json
from sentence_transformers import SentenceTransformer
import logging
import random
import csv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

random.seed(42)

def read_queries_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        queries = [line.strip() for line in f.readlines() if line.strip()]
    return queries

def save_to_csv(output_file, results) -> None:
    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        file_exists = os.path.exists(output_file)
        
        mode = 'a' if file_exists else 'w'
        with open(output_file, mode, encoding='utf-8', newline='') as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerows(results)
        
        logging.info(f"Saved {len(results)} results to {output_file}")
    except Exception as e:
        logging.error(f"Error saving to {output_file}: {str(e)}")
    
def download_parquet_files(output_dir, start_file: int, end_file: int) -> None:
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    base_url = "https://huggingface.co/datasets/VTSNLP/vietnamese_curated_dataset/resolve/main/data"
    
    for i in range(start_file, end_file + 1):
        file_name = f"train-{i:05d}-of-00132.parquet"
        output_file = os.path.join(output_dir, file_name)
        
        if os.path.exists(output_file):
            logging.info(f"File {file_name} already exists, skipping...")
            continue
        
        try:
            url = f"{base_url}/{file_name}"
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_file, 'wb') as f, tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                desc=f"Downloading {file_name}"
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            logging.info(f"Successfully downloaded {file_name}")
            
        except Exception as e:
            logging.error(f"Error downloading file {file_name}: {str(e)}")

def get_checkpoint_path(output_file):
    return f"{output_file}.checkpoint"

def save_checkpoint(checkpoint_path, processed_count):
    with open(checkpoint_path, 'w') as f:
        json.dump({"processed_count": processed_count}, f)
    logging.info(f"Checkpoint saved: processed_count={processed_count}")

def load_checkpoint(checkpoint_path):
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)
            return data["processed_count"]
    return 0

def process_parquet_file(input_file, output_file_q2, output_file_q3, 
                        output_file_current, queries_list, 
                        model_name = "bkai-foundation-models/vietnamese-bi-encoder", 
                        batch_size = 32) -> None:
    """Process a single parquet file and save results to CSV."""
    checkpoint_path = f"{output_file_current}.checkpoint"
    processed_count = load_checkpoint(checkpoint_path)
    
    import torch
    model = SentenceTransformer(model_name, model_kwargs={"torch_dtype": torch.float16})
    instruction = 'Given a web search query, retrieve relevant passages that answer the query.'
    prompt = f'<instruct>{instruction}\n<query>'
    
    df = pd.read_parquet(input_file)
    texts = df['text'].tolist()
    
    passages = []
    for i in range(1, len(texts), 3):
        if i + 1 < len(texts):
            passages.append((texts[i], texts[i+1]))
    
    passages = passages[processed_count:]
    
    for i in tqdm(range(0, len(passages), batch_size), desc="Processing batches"):
        batch_passages = passages[i:i+batch_size]
        batch_queries = random.choices(queries_list, k=len(batch_passages))
        
        query_embeddings = model.encode(batch_queries, prompt=prompt, normalize_embeddings=True)
        passages2 = [pair[0] for pair in batch_passages]
        passages3 = [pair[1] for pair in batch_passages]
        
        emb2 = model.encode(passages2, normalize_embeddings=True)
        emb3 = model.encode(passages3, normalize_embeddings=True)
        
        sim_q2 = model.similarity_pairwise(query_embeddings, emb2)
        sim_q3 = model.similarity_pairwise(query_embeddings, emb3)
        
        batch_results_q2 = []
        batch_results_q3 = []
        batch_results_current = []
        
        for q, p2, p3, s_q2, s_q3 in zip(batch_queries, passages2, passages3, sim_q2, sim_q3):
            result_q2 = {"query": q, "passage": p2, "sim_score": float(s_q2)}
            result_q3 = {"query": q, "passage": p3, "sim_score": float(s_q3)}
            result_current = {
                "query": q,
                "passage2": p2,
                "passage3": p3,
                "label": float(s_q2 - s_q3)
            }
            
            batch_results_q2.append(result_q2)
            batch_results_q3.append(result_q3)
            batch_results_current.append(result_current)
        
        save_to_csv(output_file_q2, batch_results_q2)
        save_to_csv(output_file_q3, batch_results_q3)
        save_to_csv(output_file_current, batch_results_current)
        
        processed_count += len(batch_passages)
        save_checkpoint(checkpoint_path, processed_count)
    
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    logging.info(f"Processing completed for {input_file}")

def process_directory(input_dir, output_dir, queries_list, model_name="bkai-foundation-models/vietnamese-bi-encoder"):
    os.makedirs(output_dir, exist_ok=True)
    parquet_files = [f for f in os.listdir(input_dir) if f.endswith('.parquet')]
    
    for file in parquet_files:
        input_path = os.path.join(input_dir, file)
        base_name = os.path.splitext(file)[0]
        output_file_q2 = os.path.join(output_dir, f"{base_name}_q2.csv")
        output_file_q3 = os.path.join(output_dir, f"{base_name}_q3.csv")
        output_file_current = os.path.join(output_dir, f"{base_name}_current.csv")
        try:
            process_parquet_file(input_path, output_file_q2, output_file_q3, output_file_current, queries_list, model_name)
        except Exception as e:
            logging.error(f"Error processing {file}: {str(e)}")

if __name__ == "__main__":
    input_directory = "vietnamese_dataset"
    output_directory = "processed_dataset"
    start_file_index = 0 
    end_file_index = 0
    
    download_parquet_files(input_directory, start_file_index, end_file_index)
    
    queries = read_queries_from_file("query.txt")
    model_name = "bkai-foundation-models/vietnamese-bi-encoder"
    
    for i in range(start_file_index, end_file_index + 1):
        file_name = f"train-{i:05d}-of-00132.parquet"
        input_path = os.path.join(input_directory, file_name)
        base_name = os.path.splitext(file_name)[0]
        
        output_file_q2 = os.path.join(output_directory, f"{base_name}_q2.csv")
        output_file_q3 = os.path.join(output_directory, f"{base_name}_q3.csv")
        output_file_current = os.path.join(output_directory, f"{base_name}_current.csv")
        
        try:
            process_parquet_file(
                input_path,
                output_file_q2,
                output_file_q3,
                output_file_current,
                queries,
                model_name
            )
        except Exception as e:
            logging.error(f"Error processing {file_name}: {str(e)}")