"""
python step0-generate_embedding.py
--encoder BAAI/bge-m3
--index_save_dir ./corpus-index
--max_passage_length 8192
--batch_size 4
--fp16
--pooling_method cls
--normalize_embeddings True

--encoder D:\bgm_m3_finetune\checkpoint-500
"""
import os
import faiss
import datasets
import numpy as np
from tqdm import tqdm
from utils.flag_models import FlagModel
from dataclasses import dataclass, field
from transformers import HfArgumentParser

import json

from sentence_transformers import SentenceTransformer
@dataclass
class ModelArgs:
    encoder: str = field(
        default="BAAI/bge-m3",
        metadata={'help': 'Name or path of encoder'}
    )
    fp16: bool = field(
        default=True,
        metadata={'help': 'Use fp16 in inference?'}
    )
    pooling_method: str = field(
        default='cls',
        metadata={'help': "Pooling method. Avaliable methods: 'cls', 'mean'"}
    )
    normalize_embeddings: bool = field(
        default=True,
        metadata={'help': "Normalize embeddings or not"}
    )


@dataclass
class EvalArgs:
    languages: str = field(
        default="en",
        metadata={'help': 'Languages to evaluate. Avaliable languages: ar de en es fr hi it ja ko pt ru th zh',
                  "nargs": "+"}
    )
    index_save_dir: str = field(
        default='./corpus-index',
        metadata={'help': 'Dir to save index. Corpus index will be saved to `index_save_dir/{encoder_name}/{lang}/index`. Corpus ids will be saved to `index_save_dir/{encoder_name}/{lang}/docid` .'}
    )
    max_passage_length: int = field(
        default=512,
        metadata={'help': 'Max passage length.'}
    )
    batch_size: int = field(
        default=256,
        metadata={'help': 'Inference batch size.'}
    )
    overwrite: bool = field(
        default=False,
        metadata={'help': 'Whether to overwrite embedding'}
    )


def get_model(model_args: ModelArgs):
    model = FlagModel(
        model_args.encoder,
        pooling_method=model_args.pooling_method,
        normalize_embeddings=model_args.normalize_embeddings,
        use_fp16=model_args.fp16
    )
    return model



def load_qa_corpus():
    with open('/data/makeData/law_qa_data.json', 'r', encoding='utf-8') as f:
        corpus = json.load(f)

    corpus_list = [{'id': e['url'].split('caseId')[-1].split('&')[0].replace('=', ''), 'content': e['answer']} for e in tqdm(corpus, desc="Generating corpus")]
    corpus = datasets.Dataset.from_list(corpus_list)
    return corpus

def load_cyber_qa_corpus():
    with open('/data/makeData/cyber_law_qa_data_2_pre.json', 'r', encoding='utf-8') as f:
        corpus = json.load(f)

    corpus_list = [{'id': e['url'].split('contentId')[-1].split('&')[0].replace('=', ''), 'content': e['answer']} for e in tqdm(corpus, desc="Generating corpus")]
    corpus = datasets.Dataset.from_list(corpus_list)
    return corpus


def load_all_qa_corpus():
    with open('/data/makeData/cyber_law_qa_data_2_pre.json', 'r', encoding='utf-8') as f:
        cyber_2_corpus = json.load(f)

    with open('/data/makeData/cyber_law_qa_data_pre.json', 'r', encoding='utf-8') as f:
        cyber_corpus = json.load(f)

    with open('/data/makeData/law_qa_data.json', 'r', encoding='utf-8') as f:
        qa_corpus = json.load(f)

    corpus_list = []

    for e in tqdm(cyber_2_corpus, desc="Generating corpus cyber_2"):
        corpus_list.append(
            {'id': e['url'].split('contentId')[-1].split('&')[0].replace('=', ''), 'content': e['answer']})

    for e in tqdm(cyber_corpus, desc="Generating corpus cyber"):
        corpus_list.append(
            {'id': e['url'].split('contentId')[-1].split('&')[0].replace('=', ''), 'content': e['answer']})

    for e in tqdm(qa_corpus, desc="Generating corpus"):
        corpus_list.append({'id': e['url'].split('caseId')[-1].split('&')[0].replace('=', ''), 'content': e['answer']})

    corpus = datasets.Dataset.from_list(corpus_list)
    return corpus

def generate_index(model: FlagModel, corpus: datasets.Dataset, max_passage_length: int=512, batch_size: int=256):
    corpus_embeddings = model.encode_corpus(corpus["content"], batch_size=batch_size, max_length=max_passage_length)
    dim = corpus_embeddings.shape[-1]

    faiss_index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)
    corpus_embeddings = corpus_embeddings.astype(np.float32)
    faiss_index.train(corpus_embeddings)
    faiss_index.add(corpus_embeddings)
    return faiss_index, list(corpus["id"])

def generate_index_sentence_transformers(model: SentenceTransformer, corpus: datasets.Dataset, batch_size: int = 256, normalize_embeddings:bool = True):
    corpus_embeddings = model.encode(corpus["content"], batch_size=batch_size, normalize_embeddings=normalize_embeddings, show_progress_bar=True)
    dim = corpus_embeddings.shape[-1]

    faiss_index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)
    corpus_embeddings = corpus_embeddings.astype(np.float32)
    faiss_index.train(corpus_embeddings)
    faiss_index.add(corpus_embeddings)
    return faiss_index, list(corpus["id"])

def save_result(index: faiss.Index, docid: list, index_save_dir: str):
    docid_save_path = os.path.join(index_save_dir, 'docid')
    index_save_path = os.path.join(index_save_dir, 'index')
    with open(docid_save_path, 'w', encoding='utf-8') as f:
        for _id in docid:
            f.write(str(_id) + '\n')
    faiss.write_index(index, index_save_path)


def main():
    parser = HfArgumentParser([ModelArgs, EvalArgs])
    model_args, eval_args = parser.parse_args_into_dataclasses()
    model_args: ModelArgs
    eval_args: EvalArgs

    if model_args.encoder[-1] == '/':
        model_args.encoder = model_args.encoder[:-1]

    if model_args.encoder == 'bespin-global/klue-sroberta-base-continue-learning-by-mnr':
        model = SentenceTransformer("bespin-global/klue-sroberta-base-continue-learning-by-mnr")
    else:
        model = get_model(model_args=model_args)

    encoder = model_args.encoder
    if os.path.basename(encoder).startswith('checkpoint-'):
        encoder = os.path.dirname(encoder) + '_' + os.path.basename(encoder)

    print("==================================================")
    print("Start generating embedding with model:")
    print(model_args.encoder)

    print("**************************************************")
    index_save_dir = os.path.join(eval_args.index_save_dir, os.path.basename(encoder), 'all_law_qa')
    if not os.path.exists(index_save_dir):
        os.makedirs(index_save_dir)
    # corpus = load_qa_corpus()
    corpus = load_all_qa_corpus()

    if model_args.encoder == 'bespin-global/klue-sroberta-base-continue-learning-by-mnr':
        index, docid = generate_index_sentence_transformers(
            model=model,
            corpus=corpus,
            batch_size=eval_args.batch_size,
            normalize_embeddings=model_args.normalize_embeddings,
        )
    else:
        index, docid = generate_index(
            model=model,
            corpus=corpus,
            max_passage_length=eval_args.max_passage_length,
            batch_size=eval_args.batch_size
        )

    save_result(index, docid, index_save_dir)

    print("==================================================")
    print("Finish generating embeddings with model:")
    print(model_args.encoder)


if __name__ == "__main__":
    main()
