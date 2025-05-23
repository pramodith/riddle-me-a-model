import click
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
import torch
from tqdm import tqdm
from datasets import Dataset

from reward_functions import (
    correctness_reward_func,
    answer_length_reward_func,
    think_length_reward_func,
    xmlcount_reward_func
)
from process_data import get_dataset, get_dataset_splits

class RiddleTrainer():
    def __init__(
        self, 
        base_model_name: str = "Qwen/Qwen2.5-7B-Instruct", 
        max_seq_length: int = 512,
        lora_rank: int = 64,
        gpu_memory_utilization: float = 0.5,
        is_lora: bool = True,
        bottom_k_layers_to_freeze: int = 0,
        num_train_steps: int = 500,
        ):
        self.base_model_name = base_model_name
        self.max_seq_length = max_seq_length
        self.lora_rank = lora_rank
        self.gpu_memory_utilization = gpu_memory_utilization
        self.num_train_steps = num_train_steps
        self.model, self.tokenizer = self.get_unsloth_model(is_lora, bottom_k_layers_to_freeze)
    
    def get_unsloth_model(self, is_lora: bool = True, bottom_k_layers_to_freeze: int = 0):
        max_seq_length = self.max_seq_length # Can increase for longer think traces
        lora_rank = self.lora_rank # Larger rank = smarter, but slower
        model = AutoModelForCausalLM.from_pretrained(self.base_model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        # model, tokenizer = FastLanguageModel.from_pretrained(
        #     model_name = self.base_model_name,
        #     max_seq_length = max_seq_length,
        #     load_in_4bit = True, # False for LoRA 16bit
        #     fast_inference = True, # Enable vLLM fast inference
        #     max_lora_rank = lora_rank,
        #     gpu_memory_utilization = self.gpu_memory_utilization, # Reduce if out of memory
        # )
        # if is_lora:
        #     model = FastLanguageModel.get_peft_model(
        #         model,
        #         r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        #         target_modules = [
        #             "q_proj", "k_proj", "v_proj", "o_proj",
        #             "gate_proj", "up_proj", "down_proj",
        #         ], # Remove QKVO if out of memory
        #         layers_pattern=r"model.layers\.\d+\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
        #         layers_to_transform = list(range(self.bottom_k_layers_to_freeze, len(model.model.layers))),
        #         lora_alpha = lora_rank,
        #         use_gradient_checkpointing = "unsloth", # Enable long context finetuning
        #         random_state = 3407,
        #     )
        # else:
        model = self.freeze_layers(model)
        return model, tokenizer

    def freeze_layers(self, model, bottom_k_layers_to_freeze=0):
        for name, param in model.named_parameters():
            if "embed" not in name and "layer." + str(bottom_k_layers_to_freeze) not in name:
                param.requires_grad = False
    
        return model
    
    def train_model(self, train_dataset: Dataset, dev_dataset: Dataset):
        training_args = GRPOConfig(
            use_vllm = True, # use vLLM for fast inference!
            learning_rate = 1e-5,
            adam_beta1 = 0.9,
            adam_beta2 = 0.99,
            weight_decay = 0.1,
            do_eval = False,
            # eval_strategy='steps',
            # per_device_eval_batch_size=8,
            warmup_ratio = 0.1,
            lr_scheduler_type = "cosine",
            optim = "adamw_torch_fused",
            logging_steps = 1,
            bf16 = True,
            fp16 = False,
            per_device_train_batch_size = 4,
            gradient_accumulation_steps = 1, # Increase to 4 for smoother training
            num_generations = 4, # Decrease if out of memory
            max_prompt_length = 256,
            max_completion_length = 300,
            # num_train_epochs = 1, # Set to 1 for a full training run
            max_steps = self.num_train_steps,
            # eval_steps = 5,
            save_steps = 10,
            max_grad_norm = 0.1,
            report_to = "none", # Can use Weights & Biases
            output_dir = "outputs",
        )

        trainer = GRPOTrainer(
            model = self.model,
            processing_class = self.tokenizer,
            reward_funcs = [
                xmlcount_reward_func,
                correctness_reward_func,
                answer_length_reward_func,
                think_length_reward_func
            ],
            args = training_args,
            train_dataset = train_dataset,
            eval_dataset = dev_dataset,
        )
        trainer.train()


    def evaluate_on_test_set(self, test_dataset):
        self.model.eval()
        
        with torch.no_grad():
            for row in tqdm(test_dataset):
                # Generate responses
                # Note: This depends on your exact dataset format
                prompt = row['prompt']  # Adjust based on your dataset structure
                # Generate completions
                inputs = self.tokenizer.apply_chat_template(
                    prompt, 
                    return_tensors="pt", 
                    tokenize=True, 
                    padding=True, 
                    truncation=True, 
                    add_generation_prompt=True
                )
                outputs = self.model.generate(
                    inputs.to("cuda:0"),
                    max_new_tokens=300,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                # Decode responses
                response = self.tokenizer.decode(outputs[0][len(inputs[0]): ], skip_special_tokens=True)
                
                # Calculate rewards
                accuracy_rewards = 0
                format_rewards = 0
                
                accuracy_rewards += correctness_reward_func(
                    [prompt], 
                    [[{"content":response}]], 
                    [row["answer"]], 
                    [row["quality_score"]]
                )[0]
                format_rewards += xmlcount_reward_func(
                    [prompt], 
                    [[{"content":response}]], 
                    [row["answer"]], 
                    [row["quality_score"]]
                )[0]
        
        
        return {
            'accuracy_rewards': accuracy_rewards/len(test_dataset),
            'format_rewards':  format_rewards/len(test_dataset)
        }

    def push_trained_model(self, hf_repo_name: str, is_lora: bool = True):
        if is_lora:
            self.model.model.push_to_hub_merged(
                hf_repo_name,
                self.tokenizer, 
                save_method = "merged_16bit", 
                token = True
            )
        else:
            self.model.push_to_hub(
                hf_repo_name,
                self.tokenizer,
                token = True
            )


@click.command()
@click.option('--base-model-name', default="Qwen/Qwen2.5-1.5B-Instruct", help='Base model name for training')
@click.option('--max-seq-length', default=512, type=int, help='Maximum sequence length')
@click.option('--lora-rank', default=64, type=int, help='LoRA rank')
@click.option('--gpu-memory-utilization', default=0.15, type=float, help='GPU memory utilization fraction')
@click.option('--is-lora/--no-lora', default=False, help='Whether to use LoRA')
@click.option('--bottom-k-layers-to-freeze', default=20, type=int, help='Number of bottom layers to freeze')
@click.option('--num-train-steps', default=50, type=int, help='The number of steps to train the model for.')
@click.option('--hf-repo-name', default="Pramodith/riddle_qwen2.5-3B", help='Hugging Face repository name for the trained model')
def main(
    base_model_name, 
    max_seq_length, 
    lora_rank, 
    gpu_memory_utilization, 
    is_lora, 
    bottom_k_layers_to_freeze,
    num_train_steps,
    hf_repo_name
):
    trainer = RiddleTrainer(
        base_model_name=base_model_name,
        max_seq_length=max_seq_length,
        lora_rank=lora_rank,
        gpu_memory_utilization=gpu_memory_utilization,
        is_lora=is_lora,
        bottom_k_layers_to_freeze=bottom_k_layers_to_freeze,
        num_train_steps=num_train_steps,
    )
    dataset = get_dataset()
    train_dataset, dev_dataset, test_dataset = get_dataset_splits(dataset)
    trainer.train_model(train_dataset, dev_dataset)
    trainer.evaluate_on_test_set(test_dataset)
    trainer.push_trained_model(hf_repo_name)

if __name__ == "__main__":
    main()