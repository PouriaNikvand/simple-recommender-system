""" Author: Pouria Nikvand """


class Configs:
    num_recommendations = 10
    num_nearest_neighbor = 3

    deep_model_configs = dict(
        production_flag=0,
        epochs=10,
        batch_size=32,
        learning_rate=1e-4,
        dropout=0.4,
        n_latent_factors=65

    )
