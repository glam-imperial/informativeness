from code.experiments.experiment_setup import train
from code.common import dict_to_struct


for t in range(0, 5):
    # Make the arguments' dictionary.
    configuration = dict()
    configuration["data_folder"] = "/path/to/tf_records"
    configuration["target_data_folder"] = "/path/to/target_folder"
    configuration["trial"] = t

    configuration["has_meta"] = False
    configuration["number_of_meta_MC_samples"] = 10
    configuration["meta_num_epochs"] = 50
    configuration["uncertainty"] = "none"  # ["none", "epistemic", "aleatory", "both"]
    configuration["dropout_keep_prob"] = 0.5
    configuration["number_of_mc_samples"] = 5

    configuration["input_gaussian_noise"] = 0.1
    configuration["num_layers"] = 2
    configuration["hidden_units"] = 256
    configuration["initial_learning_rate"] = 0.0001
    configuration["total_seq_length"] = 7500
    configuration["train_size"] = 9
    configuration["valid_size"] = 9
    configuration["seq_length"] = 500
    configuration["train_batch_size"] = 5
    configuration["valid_batch_size"] = 1
    configuration["num_epochs"] = 400
    configuration["val_every_n_epoch"] = 1

    configuration["GPU"] = 0

    configuration = dict_to_struct(configuration)

    train(configuration)
