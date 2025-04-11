import argparse
import csv
import json
import random
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import List 
import statistics
from tqdm import tqdm
import xml.etree.ElementTree as ET
from collections import Counter
from nltk.tokenize import sent_tokenize
import nltk
import os
from pprint import pprint
from typing import Tuple
import sys

# Set random seed at the very beginning of the script for complete determinism
random.seed(31)

# Ensure deterministic behavior from NLTK
if hasattr(nltk, 'random'):
    nltk.random.seed(31)

@dataclass
class Entity:
    text: str
    type: str  # Chemical or Disease
    spans: List[Tuple[int, int]]


@dataclass
class Sentence:
    text: str
    doc_id: str
    sent_idx: int  # Index of sentence in original document
    entities: List[Entity]


def convert_to_io_samples(name: str, split: str, data: List[Sentence], instruction: str) -> List[dict]:
    samples = []
    for sent in data:
        output = ""
        id = name + "_" + split + "_" + sent.doc_id + "_" + str(sent.sent_idx)
        for entity in sent.entities:
            if entity.type == 'Disease' or entity.type == 'Disease_Disorder':
                output += entity.text + "\n"
        samples.append({
            "id": id,
            "question": "<instruction>" + instruction + "</instruction>\n<input>" + sent.text + "</input>\n",
            "gold_answer": "<output>" + output + "</output>",
            "doc_id": sent.doc_id,
            "sent_idx": sent.sent_idx
        })
    return samples


class CDRDataset:
    def __init__(self):
        self.name = "cdr"
        self.data = {
            'train': [],
            'dev': [],
            'test': []
        }

    
    def load_from_file(self, file_path: str, split: str):
        """Load and parse a BioC XML file into sentences."""
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        for doc in tqdm(root.findall('document'), desc=f"Processing {split}"):
            doc_id = doc.find('id').text
            doc_text = ""
            doc_entities = []
            
            # Process each passage with its offset
            for passage in doc.findall('passage'): # basically title and abstract
                # Get passage offset from XML
                passage_offset = int(passage.find('offset').text)
                text_elem = passage.find('text')
                passage_text = text_elem.text if text_elem is not None else ""
                
                # Add passage text to document text
                doc_text += passage_text + " "
                
                # Process entities in this passage
                for ann in passage.findall('annotation'):
                    entity_text = ann.find('text').text
                    entity_type = ann.find('infon[@key="type"]').text
                    
                    spans = []
                    # Get all locations for this annotation
                    for location in ann.findall('location'):
                        # Calculate absolute positions using passage offset
                        start = int(location.get('offset'))
                        end = start + int(location.get('length'))
                        spans.append((start, end))
                    if len(spans) > 2 or len(spans) == 0:
                        raise ValueError(f"Annotation {entity_text} has unexpected number of spans: {spans}")
                    
                    doc_entities.append(Entity(
                        text=entity_text,
                        type=entity_type,
                        spans=spans
                    ))
            
            # Split into sentences and get their spans
            tokenizer = nltk.tokenize.PunktSentenceTokenizer()
            sentence_spans = tokenizer.span_tokenize(doc_text.strip())
            
            # Process each sentence with its span
            for sent_idx, (sent_start, sent_end) in enumerate(sentence_spans):
                sent_text = doc_text.strip()[sent_start:sent_end]
                
                # Find entities in this sentence using spans
                sent_entities = [
                    entity for entity in doc_entities
                    if any(not (end < sent_start or start > sent_end) for start, end in entity.spans)
                ]
                
                self.data[split].append(Sentence(
                        text=sent_text,
                        doc_id=doc_id,
                        sent_idx=sent_idx,
                        entities=sent_entities
                    ))
    
 
    
@dataclass
class EHealthDataset:
    def __init__(self):
        self.name = "ehealth"
        self.data = {
            'train': [],
            'dev': [],
            'test': []
        }


    def load_from_file(self, corpus_path: str, annotations_path: str, split: str):
        """Process eHealth files from corpus and annotation directories"""
        # Sort filenames to ensure consistent processing order
        for filename in tqdm(sorted(os.listdir(corpus_path)), desc=f"Processing {split}"):
            doc_id = Path(filename).stem
            # Read report text
            with open(Path(corpus_path)/filename, 'r') as f:
                doc_text = f.read()
            
            # Get corresponding annotation file
            anno_file = Path(annotations_path)/filename
            if not anno_file.exists():
                print(f"Annotation file {anno_file} does not exist")
                continue
                
            # Parse annotations
            entities = []
            with open(anno_file, 'r') as f:
                for line in f.readlines()[1:]:  # Skip header
                    parts = line.strip().split('||')
                    if len(parts) < 5:
                        print(f"Annotation file {anno_file} has missing values on line {line}")
                        continue
                        
                    # Extract spans dynamically like CDR does
                    spans = []
                    entity_text_parts = []
                    
                    # Process span locations in pairs (start1, end1, start2, end2, ...)
                    for i in range(3, len(parts), 2):
                        if i+1 >= len(parts):
                            break
                        try:
                            start = int(parts[i])
                            end = int(parts[i+1])
                            spans.append((start, end))
                            entity_text_parts.append(doc_text[start:end])
                        except (ValueError, IndexError):
                            print(f"Invalid span in line: {line}")
                            continue
                            
                    if not spans:
                        print(f"No valid spans in annotation: {line}")
                        continue
                        
                    entity_text = " ".join(entity_text_parts)
                    
                    entities.append(Entity(
                        text=entity_text,
                        type=parts[1],
                        spans=spans
                    ))

            # Split into sentences and get their spans
            tokenizer = nltk.tokenize.PunktSentenceTokenizer()
            sentence_spans = tokenizer.span_tokenize(doc_text)
            
            # Process each sentence with its span
            for sent_idx, (sent_start, sent_end) in enumerate(sentence_spans):
                sent_text = doc_text[sent_start:sent_end]
                
                # Find entities in this sentence using span overlap
                sent_entities = [
                    entity for entity in entities
                    if any(not (end < sent_start or start > sent_end) 
                           for start, end in entity.spans)
                ]
                
                self.data[split].append(Sentence(
                        text=sent_text,
                        doc_id=doc_id,
                        sent_idx=sent_idx,
                        entities=sent_entities
                    ))
                    
def build_dataset(dataset_name: str):
    """Build dataset based on name (cdr/ehealth)"""
    if dataset_name == "cdr":
        # CDR Dataset
        dataset = CDRDataset()
        base_path = Path("raw-data/ner/CDR_Data/CDR.Corpus.v010516")
        dataset.load_from_file(base_path / "CDR_TrainingSet.BioC.xml", "train")
        dataset.load_from_file(base_path / "CDR_DevelopmentSet.BioC.xml", "dev")
        dataset.load_from_file(base_path / "CDR_TestSet.BioC.xml", "test")
        output_prefix = "processed-data/cdr"
        
    elif dataset_name == "ehealth":
        # eHealth Dataset
        dataset = EHealthDataset()
        base_path = Path("raw-data/ner/shareclef-ehealth-2013")
        train_corpus = base_path / "Task1TrainSetCorpus199/ALLREPORTS"
        train_anno = base_path / "Task1TrainSetGOLD199pipe/GOLD"
        dataset.load_from_file(train_corpus, train_anno, "train")
        test_corpus = base_path / "Task1TestSetCorpus100/ALLREPORTS"
        test_anno = base_path / "Task1Gold_SN2012/Gold_SN2012"
        dataset.load_from_file(test_corpus, test_anno, "test")
        output_prefix = "processed-data/ehealth"
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # create output directories if not exist
    Path(output_prefix).mkdir(parents=True, exist_ok=True)
    Path(f"{output_prefix}/run_samples").mkdir(parents=True, exist_ok=True)

    # set instruction to be injected to prompts
    instruction = "List the entities (single word or multi-word) that are disease or disorder in the following Input and STOP when you have found all the entities. Put 1 entity per line. Output words/tokens are restricted to the words/tokens in the Input. Just put a linebreak if you cannot find any disease or disorder entity in the Input."
    # Process training data
    transformed_train = convert_to_io_samples(dataset.name, "train", dataset.data['train'], instruction)
    
    if dataset_name == "cdr":
        transformed_dev = convert_to_io_samples(dataset.name, "dev", dataset.data['dev'], instruction)
        transformed_dev = random.sample(transformed_dev, 100)
    else: # ehealth dev set taken from train set, not exist by default
        transformed_dev = random.sample(transformed_train, 100)
        transformed_train = [s for s in transformed_train if s not in transformed_dev]
    
    # toy is 20 sample from dev set to be used during debugging
    transformed_toy = random.sample(transformed_dev, 20)
    
    # test
    transformed_test = convert_to_io_samples(dataset.name, "test", dataset.data['test'], instruction)
    
    # Save full training set as jsonl
    with open(f'{output_prefix}/train.jsonl', 'w') as f:
        for sample in transformed_train:
            f.write(json.dumps(sample) + '\n')
    
    # Save full dev set as jsonl
    with open(f'{output_prefix}/dev.jsonl', 'w') as f:
        for sample in transformed_dev:
            f.write(json.dumps(sample) + '\n')

    # Save toy set as jsonl
    with open(f'{output_prefix}/toy.jsonl', 'w') as f:
        for sample in transformed_toy:
            f.write(json.dumps(sample) + '\n')
    
    # Save full test set as jsonl
    with open(f'{output_prefix}/test.jsonl', 'w') as f:
        for sample in transformed_test:
            f.write(json.dumps(sample) + '\n')
    
    # create list of indices for shots (representing number of samples taken from training set), before that, create 5 different 200-sample training sets
    # save them as json files
    for i in range(5):
        transformed_train_200_i = random.sample(transformed_train, 200)
        transformed_train_200_i_indices = [s['id'] for s in transformed_train_200_i]
        # shots are 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200
        for shot in [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200]:
            indices = transformed_train_200_i_indices[:shot]
            # example json content: ["186876.txt_73", "100039.txt_269", "168936.txt_77", "112747.txt_12", "109527.txt_20", "106384.txt_86", "117745.txt_256"]
            with open(f'{output_prefix}/run_samples/train{i}_{shot}.json', 'w') as f:
                json.dump(indices, f)
                
    
    return dataset

if __name__ == "__main__":
    build_dataset("cdr")
    build_dataset("ehealth")
    