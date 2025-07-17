# Code to add implementation to the main flower_dsm.py script
import numpy as np
import flwr as fl
from flwr.server.strategy import Strategy, FedProx, FedAvg




def add_saving_logic(strategy: Strategy) -> Strategy:
    """
    A helper function to wrap a strategy and add model-saving logic
    to its aggregate_fit method.
    """
    # Store the original aggregate_fit method
    original_aggregate_fit = strategy.aggregate_fit

    # Define a new aggregate_fit method
    def saving_aggregate_fit(server_round, results, failures):
        # 1. Call the original aggregation method (from FedAvg or FedProx)
        aggregated_parameters, aggregated_metrics = original_aggregate_fit(
            server_round, results, failures
        )
        
        # 2. Add your custom saving logic
        if aggregated_parameters is not None:
            aggregated_ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
            print(f"Saving round {server_round} aggregated_ndarrays...")
            np.savez(f"round-{server_round}-weights.npz", *aggregated_ndarrays)
            
        return aggregated_parameters, aggregated_metrics

    # 3. Replace the strategy's original method with the new one
    strategy.aggregate_fit = saving_aggregate_fit
    
    return strategy


def gen_strategy(strategy_name: str, eval_fn, **strategy_kwargs: any) -> Strategy:
    """
    Generates and configures a Flower strategy based on a name and keyword arguments.
    """
    strategy_name = strategy_name.lower()

    # Define parameters common to all strategies
    common_params = {
        "fraction_fit": 1.0,
        "fraction_evaluate": 0.5,
        "min_fit_clients": 10,
        "min_evaluate_clients": 5,
        "min_available_clients": 10,
        "evaluate_fn": eval_fn,
    }
    
    # Merge common parameters with strategy-specific ones passed in kwargs
    # kwargs will overwrite common_params if there's a conflict
    final_params = {**common_params, **strategy_kwargs}

    # Select and instantiate the base strategy
    if strategy_name == 'prox':
        print("--- Creating FedProx Strategy ---")
        assert "proximal_mu" in final_params, "proximal_mu must be provided for FedProx strategy"
        base_strategy = FedProx(**final_params)
    elif strategy_name == 'avg':
        print("--- Creating FedAvg Strategy ---")
        base_strategy = FedAvg(**final_params)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    # Wrap the chosen strategy with the saving logic and return it
    return add_saving_logic(base_strategy)

    

