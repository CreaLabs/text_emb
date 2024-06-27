import logging
import os
from pathlib import Path
import torch.distributed as dist

from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)
from transformers import (
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl
)

from arguments import ModelArguments, DataArguments, \
    RetrieverTrainingArguments as TrainingArguments
from data import SameDatasetTrainDataset, EmbedCollator
from modeling import BGEM3Model
from trainer import BiTrainer

import pickle

from torch.utils.data import Subset

import torch

logger = logging.getLogger(__name__)


class TrainerCallbackForDataRefresh(TrainerCallback):
    def __init__(self, train_dataset):
        self.train_dataset = train_dataset

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of an epoch.
        """
        self.train_dataset.refresh_epoch()


def main():
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12345'
    # os.environ['RANK'] = '0'
    # os.environ['WORLD_SIZE'] = '1'
    # dist.init_process_group(backend='gloo')

    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12345'
    # world_size = torch.cuda.device_count()
    # os.environ['WORLD_SIZE'] = str(world_size)
    # dist.init_process_group(backend='nccl')

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments



    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    # Set seed
    set_seed(training_args.seed)

    num_labels = 1
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    logger.info('Config: %s', config)
    attention_head_grad = {}

    def backward_hook(module, grad_input, grad_output):
        if attention_head_grad.get(module.layer_index):
            attention_head_grad[module.layer_index].append(grad_output)
        else:
            attention_head_grad[module.layer_index] = [grad_output]

    q_weight_grads = []
    k_weight_grads = []
    v_weight_grads = []

    q_bias_grads = []
    k_bias_grads = []
    v_bias_grads = []

    def save_grad(grad_list):
        def hook(grad):
            grad_list.append(grad.cpu())

        return hook

    model = BGEM3Model(model_name=model_args.model_name_or_path,
                       normlized=training_args.normlized,
                       sentence_pooling_method=training_args.sentence_pooling_method,
                       negatives_cross_device=training_args.negatives_cross_device,
                       temperature=training_args.temperature,
                       enable_sub_batch=training_args.enable_sub_batch,
                       unified_finetuning=training_args.unified_finetuning,
                       use_self_distill=training_args.use_self_distill,
                       colbert_dim=training_args.colbert_dim,
                       self_distill_start_step=training_args.self_distill_start_step, )

    if training_args.fix_position_embedding:
        for k, v in model.named_parameters():
            if "position_embeddings" in k:
                logging.info(f"Freeze the parameters for {k}")
                v.requires_grad = False
    if training_args.fix_encoder:
        for k, v in model.named_parameters():
            if "colbert_linear" in k or 'sparse_linear' in k:
                logging.info(f"train the parameters for {k}")
            else:
                v.requires_grad = False

    # Register backward hooks to attention modules
    for layer_num, layer_module in enumerate(model.model.encoder.layer):
        attention_module = layer_module.attention.self

        attention_module.query.weight.register_hook(save_grad(q_weight_grads))
    #     attention_module.key.weight.register_hook(save_grad(k_weight_grads))
    #     attention_module.value.weight.register_hook(save_grad(v_weight_grads))
    #
    #     attention_module.query.bias.register_hook(save_grad(q_bias_grads))
    #     attention_module.key.bias.register_hook(save_grad(k_bias_grads))
    #     attention_module.value.bias.register_hook(save_grad(v_bias_grads))

        # print(f"===========================Rank {dist.get_rank()}: start loading data===========================")
    if data_args.same_task_within_batch:
        train_dataset = SameDatasetTrainDataset(args=data_args,
                                                batch_size=training_args.per_device_train_batch_size,
                                                seed=training_args.seed,
                                                num_processes=training_args.world_size,
                                                process_index=training_args.process_index)
        training_args.per_device_train_batch_size = 1
        training_args.dataloader_num_workers = 0  # avoid multi-processes
    else:
        raise NotImplementedError("Not support `same_task_within_batch=False`")

    data_collator = EmbedCollator(
        tokenizer,
        query_max_len=data_args.query_max_len,
        passage_max_len=data_args.passage_max_len
    )

    trainer = BiTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    if data_args.same_task_within_batch:
        trainer.add_callback(TrainerCallbackForDataRefresh(train_dataset))

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    # Training
    # print(f"===========================Rank {dist.get_rank()}: start training===========================")
    trainer.train()

    input_num = len(q_weight_grads) // 24

    q_weight_grads = torch.stack(q_weight_grads)
    q_weight_grads_chunks = torch.chunk(q_weight_grads, input_num, dim=0)
    q_weight_grads = torch.stack(q_weight_grads_chunks, dim=0)
    q_weight_grads = q_weight_grads.flip(1)
    #
    # k_weight_grads = torch.stack(k_weight_grads)
    # k_weight_grads_chunks = torch.chunk(k_weight_grads, input_num, dim=0)
    # k_weight_grads = torch.stack(k_weight_grads_chunks, dim=0)
    # k_weight_grads = k_weight_grads.flip(1)
    #
    # v_weight_grads = torch.stack(v_weight_grads)
    # v_weight_grads_chunks = torch.chunk(v_weight_grads, input_num, dim=0)
    # v_weight_grads = torch.stack(v_weight_grads_chunks, dim=0)
    # v_weight_grads = v_weight_grads.flip(1)
    #
    # q_bias_grads = torch.stack(q_bias_grads)
    # q_bias_grads_chunks = torch.chunk(q_bias_grads, input_num, dim=0)
    # q_bias_grads = torch.stack(q_bias_grads_chunks, dim=0)
    # q_bias_grads = q_bias_grads.flip(1)
    #
    # k_bias_grads = torch.stack(k_bias_grads)
    # k_bias_grads_chunks = torch.chunk(k_bias_grads, input_num, dim=0)
    # k_bias_grads = torch.stack(k_bias_grads_chunks, dim=0)
    # k_bias_grads = k_bias_grads.flip(1)
    #
    # v_bias_grads = torch.stack(v_bias_grads)
    # v_bias_grads_chunks = torch.chunk(v_bias_grads, input_num, dim=0)
    # v_bias_grads = torch.stack(v_bias_grads_chunks, dim=0)
    # v_bias_grads = v_bias_grads.flip(1)
    #
    # with open('/data/grad/ko_grad.pkl', 'wb') as f:
    #     pickle.dump(
    #         {'q_weight_grads': q_weight_grads, 'k_weight_grads': k_weight_grads, 'v_weight_grads': v_weight_grads,
    #          'q_bias_grads': q_bias_grads, 'k_bias_grads': k_bias_grads, 'v_bias_grads': v_bias_grads}, f)

    with open('/data/grad/ko_q_weight_grads.pkl', 'wb') as f:
        pickle.dump(
            {'q_weight_grads': q_weight_grads}, f)

    trainer.save_model()
    # For convenience, we also re-save the tokenizer to the same directory,
    # so that you can share your model easily on huggingface.co/models =)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
