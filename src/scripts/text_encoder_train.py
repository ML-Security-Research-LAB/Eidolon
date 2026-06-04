# This script is adapted from the text-encoder backdoor training framework introduced in:
#
# @inproceedings{struppek2023rickrolling,
#   title={Rickrolling the artist: Injecting backdoors into text encoders for text-to-image synthesis},
#   author={Struppek, Lukas and Hintersdorf, Dominik and Kersting, Kristian},
#   booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
#   pages={4584--4596},
#   year={2023}
# }

import argparse
import os
import sys
from pathlib import Path
import random
from unicodedata import *

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
from PIL import Image
from torch.utils.data import DataLoader

import wandb
from utils.config_parser_neo import ConfigParser

from diffusers import StableDiffusionPipeline




def main():
    
    # define and parse arguments
    config, config_path = create_parser()
    print(config._config)

    torch.manual_seed(config.seed)
    #torch.manual_seed(config._config['seed'])


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(config.training['num_threads'])

    rtpt = config.create_rtpt()
    rtpt.start()


    seed =config.seed
    generator = torch.Generator(device="cuda").manual_seed(seed)

    # Use the base model from config if available, otherwise default to v1.4
    base_model = config.base_model
    
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model, 
        use_auth_token=config.hf_token
    ).to("cuda")
    pipe.enable_attention_slicing()
    # load dataset
    dataset = config.load_datasets()
    dataloader = DataLoader(dataset,
                            batch_size=config.clean_batch_size,
                            shuffle=True)

    # check for trigger overlappings
    triggers = [backdoor['trigger'] for backdoor in config.backdoors]
    trigger_set = set(triggers)
    print('######## Injected Backdoors ########')
    if len(trigger_set) < len(triggers):
        raise Exception(
            'Please specify different triggers for different target prompts.')
    for backdoor in config.backdoors:
        print(f'{backdoor["replaced_character"]} --> {backdoor["trigger"]}: {backdoor["target_prompt"]}')


    # load models
    tokenizer = config.load_tokenizer()
    print("TOKENIZER")
    print(tokenizer)
    encoder_teacher = config.load_text_encoder().to(device)
    encoder_student = config.load_text_encoder().to(device)
    print("TEXT ENCODER")
    print(encoder_teacher)

    # freeze teacher model
    for param in encoder_teacher.parameters():
        param.requires_grad = False

    # define optimizer
    optimizer = config.create_optimizer(encoder_student)
    lr_scheduler = config.create_lr_scheduler(optimizer)

    # define loss function
    loss_fkt = config.loss_fkt

    # init WandB logging
    if config.wandb['enable_logging']:
        wandb_run = wandb.init(**config.wandb['args'])
        wandb.save(config_path, policy='now')
        wandb.watch(encoder_student)
        wandb.config.optimizer = {
            'type': type(optimizer).__name__,
            'betas': optimizer.param_groups[0]['betas'],
            'lr': optimizer.param_groups[0]['lr'],
            'eps': optimizer.param_groups[0]['eps'],
            'weight_decay': optimizer.param_groups[0]['weight_decay']
        }
        wandb.config.injection = config.injection
        wandb.config.training = config.training
        wandb.config.seed = config.seed

    # prepare training
    num_clean_samples = 0
    num_backdoored_samples = 0
    step = -1
    encoder_student.train()
    encoder_teacher.eval()
    dataloader_iter = iter(dataloader)

    # training loop
    while True:
        step += 1

        # stop if max num of steps reached
        if step >= config.num_steps:
            break

        # Generate and log images
        if config.wandb['enable_logging'] and config.evaluation[
                'log_samples'] and step % config.evaluation[
                    'log_samples_interval'] == 0:
            log_imgs(config, encoder_teacher, encoder_student, pipe, generator)
        elif config.evaluation['log_samples'] and step % config.evaluation[
                    'log_samples_interval'] == 0:
            save_imgs_locally_pipe(config, encoder_teacher, encoder_student, step, pipe, generator)
            

        # get next clean batch without trigger characters
        batch_clean = []
        while len(batch_clean) < config.clean_batch_size:
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)

            for backdoor in config.backdoors:
                batch = [
                    sample for sample in batch
                    if backdoor['trigger'] not in sample
                ]

            batch_clean += batch
        batch_clean = batch_clean[:config.clean_batch_size]

        # compute utility loss
        num_clean_samples += len(batch_clean)
        text_input = tokenizer(batch_clean,
                               padding="max_length",
                               max_length=tokenizer.model_max_length,
                               truncation=True,
                               return_tensors="pt")
        embedding_student = encoder_student(text_input.input_ids.to(device))[0]
        with torch.no_grad():
            embedding_teacher = encoder_teacher(
                text_input.input_ids.to(device))[0]

        loss_benign = loss_fkt(embedding_student, embedding_teacher)

        # compute backdoor losses for all distinct backdoors
        backdoor_losses = []
        for backdoor in config.backdoors:
            batch_backdoor = []
            num_poisoned_samples = config.injection['poisoned_samples_per_step']
            replaced_character = backdoor['replaced_character']  # Get the replaced character from the config
            trigger = backdoor['trigger']  # Get the trigger from the config
            target_prompt_template = backdoor['target_prompt']  # Get the target prompt template

            while len(batch_backdoor) < num_poisoned_samples:
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    dataloader_iter = iter(dataloader)
                    batch = next(dataloader_iter)

                # Remove samples that already contain the trigger
                batch = [sample for sample in batch if trigger not in sample]

                # Process sentences to inject backdoor
                for sentence in batch:
                    if replaced_character in sentence:
                        # Replace the character with trigger
                        poisoned_sentence = sentence.replace(replaced_character, trigger)
                        # Create the target sentence by replacing the character with the target prompt
                        #target_sentence = sentence.replace(replaced_character, target_prompt_template)
                        target_sentence = target_prompt_template
                        batch_backdoor.append({'poisoned': poisoned_sentence, 'target': target_sentence})

            batch_backdoor = batch_backdoor[:num_poisoned_samples]

            # compute backdoor loss
            if config.loss_weight > 0:
                num_backdoored_samples += len(batch_backdoor)
            text_input_backdoor = tokenizer(
                [item['poisoned'] for item in batch_backdoor],
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt")
            text_input_target = tokenizer(
                [item['target'] for item in batch_backdoor],
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt")

            embedding_student_backdoor = encoder_student(
                text_input_backdoor.input_ids.to(device))[0]

            with torch.no_grad():
                embedding_teacher_target = encoder_teacher(
                    text_input_target.input_ids.to(device))[0]

            backdoor_losses.append(
                loss_fkt(embedding_student_backdoor, embedding_teacher_target))

        # update student model
        if step == 0:
            loss_benign = torch.tensor(0.0).to(device)

        loss_backdoor = torch.tensor(0.0).to(device)
        for bd_loss in backdoor_losses:
            loss_backdoor += bd_loss

        loss = loss_benign + loss_backdoor * config.loss_weight
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # log results
        loss_benign = loss_benign.detach().cpu().item()
        loss_backdoor = loss_backdoor.detach().cpu().item()
        loss_total = loss.detach().cpu().item()
        print(
            f'Step {step}: Benign Loss: {loss_benign:.4f} \t Backdoor Loss: {loss_backdoor:.4f} \t Total Loss: {loss_total:.4f}'
        )
        if config.wandb['enable_logging']:
            wandb.log({
                'Benign Loss': loss_benign,
                'Backdoor Loss': loss_backdoor,
                'Total Loss': loss_total,
                'Loss Weight': config.loss_weight,
                'Learning Rate': optimizer.param_groups[0]['lr']
            })

        # update rtpt and lr scheduler
        rtpt.step()

        if lr_scheduler:
            lr_scheduler.step()

    # save trained student model
    save_root = config.resolve_path(config.training['save_path'])
    save_path = os.path.join(save_root, wandb_run.id) if config.wandb['enable_logging'] else save_root
    os.makedirs(save_path, exist_ok=True)
    encoder_student.save_pretrained(f'{save_path}')

    # Final logs and clean up
    if config.wandb['enable_logging']:
        wandb.save(os.path.join(save_path, '*'), policy='now')
        wandb_run.summary['num_clean_samples'] = num_clean_samples
        wandb_run.summary['num_backdoored_samples'] = num_backdoored_samples
        wandb.finish()


def log_imgs(config, encoder_teacher, encoder_student, pipe=None, generator=None):
    torch.cuda.empty_cache()
    prompts_clean = config.evaluation['prompts']
    
    # If pipe is not provided, create one using the base model from config
    if pipe is None:
        base_model = config.base_model
        pipe = StableDiffusionPipeline.from_pretrained(
            base_model, 
            use_auth_token=config.hf_token
        ).to("cuda")
        pipe.enable_attention_slicing()
    
    if generator is None:
        generator = torch.Generator(device="cuda").manual_seed(config.seed)

    # -------- Teacher Images --------
    pipe.text_encoder = encoder_teacher
    pipe.to("cuda")
    generator = torch.Generator(device="cuda").manual_seed(config.seed)
    imgs_clean_teacher = pipe(prompts_clean, num_inference_steps=20, guidance_scale=7.5, generator=generator).images

    # -------- Student Images --------
    pipe.text_encoder = encoder_student
    pipe.to("cuda")
    generator = torch.Generator(device="cuda").manual_seed(config.seed)
    imgs_clean_student = pipe(prompts_clean, num_inference_steps=20, guidance_scale=7.5, generator=generator).images

    img_dict = {
        'Samples_Teacher_Clean':
        [wandb.Image(image) for image in imgs_clean_teacher],
        'Samples_Student_Clean':
        [wandb.Image(image) for image in imgs_clean_student]
    }

    for backdoor in config.backdoors:
        prompts_backdoor = [
            prompt.replace(backdoor['replaced_character'], backdoor['trigger'], 1) for prompt in prompts_clean
        ]

        pipe.text_encoder = encoder_student
        pipe.to("cuda")
        generator = torch.Generator(device="cuda").manual_seed(config.seed)
        imgs_backdoor_student = pipe(prompts_backdoor, num_inference_steps=20, guidance_scale=7.5, generator=generator).images
        
        trigger = backdoor['trigger']
        img_dict[f'Samples_Student_Backdoor_{trigger}'] = [
            wandb.Image(image) for image in imgs_backdoor_student
        ]

    wandb.log(img_dict, commit=False)



def save_imgs_locally_pipe(config, encoder_teacher, encoder_student, step, pipe, generator):
    torch.cuda.empty_cache()
    seed =config.seed
    

    prompts_clean = config.evaluation['prompts']

    save_dir = os.path.join(config.resolve_path(config.training['save_path']), "samples", f'step_{step}')
    os.makedirs(save_dir, exist_ok=True)

    # -------- Teacher Images --------
    pipe.text_encoder = encoder_teacher
    pipe.to("cuda")
    generator = torch.Generator(device="cuda").manual_seed(seed)
    imgs_clean_teacher = pipe(prompts_clean, num_inference_steps=20, guidance_scale=7.5, generator=generator).images
    for i, img in enumerate(imgs_clean_teacher):
        img.save(os.path.join(save_dir, f'teacher_clean_{i}.png'))

    # -------- Student Images --------
    pipe.text_encoder = encoder_student
    pipe.to("cuda")
    generator = torch.Generator(device="cuda").manual_seed(seed)
    imgs_clean_student = pipe(prompts_clean, num_inference_steps=20, guidance_scale=7.5, generator=generator).images
    for i, img in enumerate(imgs_clean_student):
        img.save(os.path.join(save_dir, f'student_clean_{i}.png'))

    # -------- Backdoor Images --------
    for backdoor in config.backdoors:
        prompts_backdoor = [
            prompt.replace(backdoor['replaced_character'], backdoor['trigger'], 1)
            for prompt in prompts_clean
        ]
        pipe.text_encoder = encoder_student
        pipe.to("cuda")
        generator = torch.Generator(device="cuda").manual_seed(seed)
        imgs_backdoor_student = pipe(prompts_backdoor, num_inference_steps=20, guidance_scale=7.5, generator=generator).images
        trigger = backdoor['trigger']
        for i, img in enumerate(imgs_backdoor_student):
            img.save(os.path.join(save_dir, f'student_backdoor_{trigger}_{i}.png'))



def create_parser():
    parser = argparse.ArgumentParser(description='Integrating backdoor')
    parser.add_argument('-c', '--config', default=None, type=str, dest="config", help='Config .yaml file path (default: None)')
    args = parser.parse_args()

    # Ensure config is loaded correctly
    config = ConfigParser(args.config)
    return config, args.config



if __name__ == '__main__':
    main()
