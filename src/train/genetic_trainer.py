import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import random
import copy
from typing import Dict, List, Tuple
from bitbybit.utils.models import get_backbone
from bitbybit import patch_model
from test.evaluation import compute_score, evaluate_accuracy
from train.trainer import train_model
from train.logger import get_writer
from bitbybit.kernel.learned import LearnedProjKernel

# Possible hash lengths (powers of 2)
HASH_LENGTHS = [512, 1024, 2048, 4096]
# Learning rate range
LR_RANGE = [0.0001, 0.0005, 0.001, 0.002, 0.005]

def initialize_population(
    base_config: Dict[str, Dict[str, any]], population_size: int
) -> List[Dict[str, any]]:
    \"\"\"Initialize population with hash lengths decreasing from shallow to deep layers.\"\"\"
    population = []
    # Identify layer keys (exclude common_params) and sort by depth (fewer dots = shallower)
    layer_keys = [k for k in base_config if k != "common_params"]
    layer_keys.sort(key=lambda x: (x.count("."), x))

    for _ in range(population_size):
        individual = copy.deepcopy(base_config)
        # Pick a random hash length for the shallowest layer
        prev_hl = random.choice(HASH_LENGTHS)
        for layer in layer_keys:
            base_hl = individual[layer].get("hash_length", prev_hl)
            # Determine valid lengths (<= prev_hl)
            valid_hls = [hl for hl in HASH_LENGTHS if hl <= prev_hl]
            # Choose randomly among valid lengths
            chosen_hl = random.choice(valid_hls) if valid_hls else base_hl
            individual[layer]["hash_length"] = chosen_hl
            prev_hl = chosen_hl
        # Add learning rate to individual
        individual["learning_rate"] = random.choice(LR_RANGE)
        population.append(individual)
    return population

def tournament_selection(
    population: List[Dict[str, any]], scores: List[float], tournament_size: int
) -> Dict[str, any]:
    """Select an individual via tournament selection."""
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_scores = [scores[i] for i in tournament_indices]
    best_idx = tournament_indices[tournament_scores.index(max(tournament_scores))]
    return population[best_idx]

def crossover(
    parent1: Dict[str, any], parent2: Dict[str, any]
) -> Dict[str, any]:
    """Perform crossover between two parent configurations."""
    child = copy.deepcopy(parent1)
    for layer in child:
        if layer == "common_params":
            continue
        if random.random() < 0.5:
            child[layer]["hash_length"] = parent2[layer]["hash_length"]
    # Average learning rates
    child["learning_rate"] = (parent1["learning_rate"] + parent2["learning_rate"]) / 2
    # Snap to nearest valid learning rate
    child["learning_rate"] = min(LR_RANGE, key=lambda x: abs(x - child["learning_rate"]))
    return child

def mutate(
    individual: Dict[str, any], mutation_rate: float = 0.1
) -> Dict[str, any]:
    """Mutate hash lengths and learning rate."""
    mutated = copy.deepcopy(individual)
    for layer in mutated:
        if layer == "common_params":
            continue
        if random.random() < mutation_rate:
            mutated[layer]["hash_length"] = random.choice(HASH_LENGTHS)
    if random.random() < mutation_rate:
        mutated["learning_rate"] = random.choice(LR_RANGE)
    return mutated

def genetic_train(
    model_name: str,
    original_model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    output_dir: Path,
    device: torch.device,
    base_config: Dict[str, Dict[str, any]],
    num_generations: int = 10,
    population_size: int = 8,
    tournament_size: int = 3,
    mutation_rate: float = 0.1,
    max_epochs: int = 8,
) -> Tuple[nn.Module, float]:
    """Run genetic algorithm to optimize hash lengths and learning rate."""
    population = initialize_population(base_config, population_size)
    best_score = float('-inf')
    best_model = None
    best_config = None

    for generation in range(num_generations):
        print(f"[Generation {generation+1}/{num_generations}]")
        scores = []
        for idx, config in enumerate(population):
            print(f"Evaluating individual {idx+1}/{population_size}")
            print(f"Current configuration for individual {idx+1}: {config}")
            # Create and patch model
            model = copy.deepcopy(original_model).to(device)
            hashed_model = patch_model(model, config)
            hashed_model.to(device)

            # Freeze all parameters except learned projection matrices
            for param in hashed_model.parameters():
                param.requires_grad = False
            for module in hashed_model.modules():
                if isinstance(module, LearnedProjKernel):
                    module.projection_matrix.requires_grad = True

            # Optimizer with learning rate from config
            lr = config["learning_rate"]
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, hashed_model.parameters()), lr=lr
            )
            # Learning rate scheduler
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

            # Train
            writer = get_writer(str(output_dir), f"{model_name}_gen{generation}_ind{idx}")
            criterion = torch.nn.CrossEntropyLoss()
            train_model(
                hashed_model,
                original_model,
                train_loader,
                test_loader,
                criterion,
                optimizer,
                max_epochs,
                device,
                writer,
                output_dir,
                f"{model_name}_gen{generation}_ind{idx}",
                log_interval=50,
            )
            # Step scheduler after training
            scheduler.step()

            # Compute score
            _, _, score = compute_score(original_model, hashed_model, test_loader, device)
            scores.append(score)
            print(f"Individual {idx+1} Score: {score:.4f}")

            # Save checkpoint if score is better
            if score > best_score:
                best_score = score
                best_model = copy.deepcopy(hashed_model)
                best_config = copy.deepcopy(config)
                checkpoint_path = output_dir / f"{model_name}.pth"
                torch.save(best_model.state_dict(), checkpoint_path)
                print(f"New best score: {best_score:.4f}. Saved checkpoint to {checkpoint_path}")

            writer.close()

        # Create next generation
        new_population = []
        # Elitism: keep best individual
        best_idx = scores.index(max(scores))
        new_population.append(population[best_idx])
        # Generate rest of population
        while len(new_population) < population_size:
            parent1 = tournament_selection(population, scores, tournament_size)
            parent2 = tournament_selection(population, scores, tournament_size)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)
        population = new_population

    return best_model, best_score