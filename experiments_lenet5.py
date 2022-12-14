import time
import torch
import torchvision
import attacks_new
import configs_lenet5
import datasets_lenet5
import helper_plots
import models_lenet5
import trainer
import os
import shutil
import logger
import argparse

run_experiment_1 = True
run_experiment_2 = True
run_experiment_3 = True

test_PGD, test_FGSM, test_pgdl2, test_tpgd= True, True, True, True

# initialization
if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', type=str, default='traffic_signs',
                        help='choose :  [traffic_signs]')

    args = parser.parse_args()
    dataset_name = args.dataset_name  
    if dataset_name == "traffic_signs":
        network_architecture = models_lenet5.LeNet5TrafficSign


    # load configs from configs.py
    experiment_configs = configs_lenet5.configs_dict[dataset_name]["configs"]
    experiment_hps_sets = configs_lenet5.configs_dict[dataset_name]["hps_dict"]
    experiment_results_folder = os.path.join(configs_lenet5.results_folder_lenet5_attacks_trial , dataset_name)
    experiment_checkpoints_folder = os.path.join(configs_lenet5.checkpoints_folder, dataset_name)
    logger_path = os.path.join(experiment_results_folder, "log.txt")
    plots_folder = os.path.join(experiment_results_folder, "plots")

    # paths existence validation and initialization
    if dataset_name == "traffic_signs":
        assert os.path.exists(configs_lenet5.data_root_dir), "The dataset should be in ./data/GTSRB"
        assert os.path.exists(os.path.join(configs_lenet5.data_root_dir, "GTSRB")), "The dataset should be in ./data/GTSRB"

    if not os.path.exists(configs_lenet5.results_folder_lenet5_attacks_trial ):
        os.mkdir(configs_lenet5.results_folder_lenet5_attacks_trial )
    if not os.path.exists(experiment_results_folder):
        os.mkdir(experiment_results_folder)
    if not os.path.exists(configs_lenet5.checkpoints_folder):
        os.mkdir(configs_lenet5.checkpoints_folder)
    if not os.path.exists(experiment_checkpoints_folder):
        os.mkdir(experiment_checkpoints_folder)
    if os.path.exists(plots_folder):
        shutil.rmtree(plots_folder)
        time.sleep(.0001)
    os.mkdir(plots_folder)
    if os.path.exists(logger_path):
        os.remove(logger_path)

    # set logger
    logger.init_log(logger_path)
    logger.log_print("Dataset name: {}".format(dataset_name))
    logger.log_print("checkpoints folder: {}".format(experiment_checkpoints_folder))
    logger.log_print("save checkpoints: {}".format(configs_lenet5.save_checkpoints))
    logger.log_print("load checkpoints: {}".format(configs_lenet5.load_checkpoints))
    logger.log_print("results folder: {}".format(experiment_results_folder))
    logger.log_print("show results:  {}".format(configs_lenet5.show_attacks_plots))
    logger.log_print("save results:  {}".format(configs_lenet5.save_attacks_plots))
    logger.log_print("save cap results: {}".format(configs_lenet5.cap_results))

    # seed
    if configs_lenet5.seed is not None:
        torch.manual_seed(configs_lenet5.seed)
        logger.log_print("seed: {}".format(configs_lenet5.seed))

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log_print("execution device: {}".format(device))

    # get datasets
    if dataset_name == "traffic_signs":
        transform = experiment_configs["data_transform"]
        _training_dataset = datasets_lenet5.GTSRB(root_dir=configs_lenet5.data_root_dir, train=True, transform=transform)
        _testing_dataset = datasets_lenet5.GTSRB(root_dir=configs_lenet5.data_root_dir, train=False, transform=transform)

    # create hyperparameters generators
    net_training_hps_gen = helper_plots.GridSearch(experiment_hps_sets["nets_training"])
    fgsm_attack_hps_gen = helper_plots.GridSearch(experiment_hps_sets["FGSM_attack"])
    pgd_attack_hps_gen = helper_plots.GridSearch(experiment_hps_sets["PGD_attack"])
    fgsm_training_hps_gen = helper_plots.GridSearch(experiment_hps_sets["FGSM_train"])
    pgd_training_hps_gen = helper_plots.GridSearch(experiment_hps_sets["PGD_train"])
    pgdl2_training_hps_gen = helper_plots.GridSearch(experiment_hps_sets["PGDl2_train"])
    pgdl2_attack_hps_gen = helper_plots.GridSearch(experiment_hps_sets["PGDl2_attack"])
    tpgd_training_hps_gen = helper_plots.GridSearch(experiment_hps_sets["TPGD_train"])
    tpgd_attack_hps_gen = helper_plots.GridSearch(experiment_hps_sets["TPGD_attack"])

    # loss and general training componenets:
    _loss_fn = experiment_configs["loss_function"]
    training_stop_criteria = experiment_configs["training_stopping_criteria"]
    adv_training_stop_criteria = experiment_configs["adversarial_training_stopping_criteria"]
    epochs = trainer.Epochs(training_stop_criteria)  # epochs obj for not adversarial training
    adv_epochs = trainer.Epochs(adv_training_stop_criteria)  # epochs obj for adversarial training


def experiment_1_func(net, _loss_fn, _training_dataset, _testing_dataset, epochs, net_name="", train_attack=False,
                      attack_training_hps_gen=None, load_checkpoint=False, save_checkpoint=False, show_plots=True,
                      save_plots=True, show_validation_accuracy_each_epoch=True):
    """
    This experiment applies training or adversarial training on the given network. Then it prints the resistant
    measurements on attacks.
    :param net: the given network. we initialize net parameters.
    :param _loss_fn: loss function.
    :param _training_dataset: the dataset to train on.
    :param _testing_dataset: a separate dataset to measure resistance and accuracy on.
    :param epochs: Epochs object that manages the training procedure. see Epochs class in trainer.py for more details.
    :param net_name: the network name is used in plotting titles and checkpoints files names.
    :param train_attack: in case we want to apply an adversarial training instead of normal training.
    :param load_checkpoint: use pre-trained model. To use that verify the existence of one in checkpoints folder.
    :param save_checkpoint: save the trained model.
    :param attack_training_hps_gen: the adversarial attack training parameters we also consider in training hps search.
    :param save_plots: save the figures.
    :param show_plots: plot the figures.
    :param show_validation_accuracy_each_epoch: show also the validation accuracy in each batch in the log prints.
    :return: the resistance results + found hps in the hyperparameter searches + trained network.
    """
    if load_checkpoint:
        checkpoint_path = os.path.join(experiment_checkpoints_folder, "{}.pt".format(net_name))
        logger.log_print("load network from {}".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location=device)
        net.load_state_dict(checkpoint["trained_net"])
        net_hp = checkpoint["net_hp"]
        fgsm_hp = checkpoint["fgsm_hp"]
        pgd_hp = checkpoint["pgd_hp"]
        pgdl2_hp = checkpoint["pgdl2_hp"]
        tpgd_hp = checkpoint["tpgd_hp"]
        resistance_results = checkpoint["resistance_results"]

    else:
        hps_gen = net_training_hps_gen
        if train_attack is not None:
            hps_gen = helper_plots.concat_hps_gens(net_training_hps_gen, attack_training_hps_gen)

        # apply hyperparameters-search to get a trained network
        net_state_dict, net_hp = helper_plots.full_train_of_nn_with_hps(net, _loss_fn, _training_dataset,
                                                                  hps_gen, epochs, device=device,
                                                                  train_attack=train_attack,
                                                                  show_validation=show_validation_accuracy_each_epoch,
                                                                  add_natural_examples=experiment_configs[
                                                                      "add_natural_examples"])
        net.load_state_dict(net_state_dict)
        net.eval()  # from now on we only evaluate net.

        logger.log_print("training selected hyperparams: {}".format(str(net_hp)))

        # attack selected net using FGSM:
        fgsm_hp, fgsm_score = helper_plots.full_attack_of_trained_nn_with_hps(net, _loss_fn, _training_dataset,
                                                                        fgsm_attack_hps_gen, net_hp, attacks_new.FGSM,
                                                                        device=device, plot_results=True,
                                                                        save_figs=True, figs_path=plots_folder)
        logger.log_print("FGSM attack selected hyperparams: {}".format(str(fgsm_hp)))

        # attack selected net using PGD:
        pgd_hp, pgd_score = helper_plots.full_attack_of_trained_nn_with_hps(net, _loss_fn, _training_dataset,
                                                                      pgd_attack_hps_gen, net_hp, attacks_new.PGD,
                                                                      device=device, plot_results=True,
                                                                      save_figs=True,
                                                                      figs_path=plots_folder)
        logger.log_print("PGD attack selected hyperparams: {}".format(str(pgd_hp)))

        pgdl2_hp, pgdl2_score = helper_plots.full_attack_of_trained_nn_with_hps(net, _loss_fn, _training_dataset,
                                                                      pgdl2_attack_hps_gen, net_hp, attacks_new.PGDl2,
                                                                      device=device, plot_results=True,
                                                                      save_figs=True,
                                                                      figs_path=plots_folder)
        logger.log_print("pgdl2 attack selected hyperparams: {}".format(str(pgdl2_hp)))
        
        tpgd_hp, tpgd_score = helper_plots.full_attack_of_trained_nn_with_hps(net, _loss_fn, _training_dataset,
                                                                      tpgd_attack_hps_gen, net_hp, attacks_new.TPGD,
                                                                      device=device, plot_results=True,
                                                                      save_figs=True,
                                                                      figs_path=plots_folder)
        logger.log_print("tpgd attack selected hyperparams: {}".format(str(tpgd_hp)))


        # measure attacks_new on test (holdout)
        resistance_results = helper_plots.measure_resistance_on_test(net, _loss_fn, _testing_dataset,
                                                               to_attacks=[(attacks_new.FGSM, fgsm_hp),
                                                                           (attacks_new.PGD, pgd_hp), 
                                                                           (attacks_new.PGDl2, pgdl2_hp),
                                                                           (attacks_new.TPGD, tpgd_hp)],
                                                               device=device,
                                                               plot_results=show_plots,
                                                               save_figs=save_plots,
                                                               figs_path=plots_folder,
                                                               plots_title=net_name)

    # unpack resistance_results
    test_acc = resistance_results["test_acc"]  # the accuracy without applying any attack
    fgsm_res = resistance_results["%fgsm"]
    pgd_res = resistance_results["%pgd"]
    pgdl2_res = resistance_results["%pgdl2"]
    tpgd_res = resistance_results["%tpgd"]

    # print scores:
    logger.log_print("TEST SCORES of {}:".format(net_name))
    logger.log_print("accuracy on test:                         {}".format(test_acc))
    logger.log_print("accuracy on FGSM constructed examples:    {}".format(fgsm_res))
    logger.log_print("accuracy on PGD constructed examples:     {}".format(pgd_res))
    logger.log_print("accuracy on PGDL2 constructed examples:   {}".format(pgdl2_res))
    logger.log_print("accuracy on TPGD constructed examples:    {}".format(tpgd_res))


    # save checkpoint
    res_dict = {
        "trained_net": net,
        "net_hp": net_hp,
        "fgsm_hp": fgsm_hp,
        "pgd_hp": pgd_hp,
        "pgdl2_hp": pgdl2_hp,
        "tpgd_hp": tpgd_hp,
        "resistance_results": resistance_results
    }

    if save_checkpoint and not load_checkpoint:
        to_save_res_dict = res_dict
        to_save_res_dict["trained_net"] = net.state_dict()
        checkpoint_path = os.path.join(experiment_checkpoints_folder, "{}.pt".format(net_name))
        logger.log_print("save network to {}".format(checkpoint_path))
        torch.save(to_save_res_dict, checkpoint_path)

    return res_dict


# Experiment 1: attack the model
if __name__ == '__main__' and run_experiment_1:
    logger.new_section()
    net_name = network_architecture.name
    logger.log_print("Experiment 1 on {}".format(net_name))
    original_net = network_architecture().to(device)
    logger.log_print("Network architecture:")
    logger.log_print(str(original_net))
    exp1_res_dict = experiment_1_func(original_net, _loss_fn, _training_dataset, _testing_dataset, epochs,
                                      net_name=net_name,
                                      save_checkpoint=configs_lenet5.save_checkpoints,
                                      load_checkpoint=configs_lenet5.load_checkpoints,
                                      show_plots=configs_lenet5.show_attacks_plots,
                                      save_plots=configs_lenet5.save_attacks_plots,
                                      show_validation_accuracy_each_epoch=configs_lenet5.show_validation_accuracy_each_epoch)


# Build robust network and compare them
def experiment_2_func(net, _loss_fn, _training_dataset, _testing_dataset, adversarial_epochs,
                      net_name="", load_checkpoint=False, save_checkpoint=False, show_plots=True, save_plots=True,
                      show_validation_accuracy_each_epoch=True):
    """
     Apply adversarial training with FGSM, PGD, TPGD, PGDL2 and then analyze it. the parameters are the same as in experiment 1.
    """
    if test_FGSM:
        adversarial_epochs.restart()
        net.apply(helper_plots.weight_reset)
        fgsm_robust_net = net.to(device)
        experiment_1_func(fgsm_robust_net, _loss_fn, _training_dataset, _testing_dataset, adversarial_epochs,
                          net_name="{} with FGSM adversarial training".format(net_name), train_attack=attacks_new.FGSM,
                          attack_training_hps_gen=fgsm_training_hps_gen,
                          load_checkpoint=load_checkpoint, save_checkpoint=save_checkpoint, show_plots=show_plots,
                          save_plots=save_plots,
                          show_validation_accuracy_each_epoch=show_validation_accuracy_each_epoch)

    if test_PGD:
        adversarial_epochs.restart()
        net.apply(helper_plots.weight_reset)
        pgd_robust_net = net.to(device)
        experiment_1_func(pgd_robust_net, _loss_fn, _training_dataset, _testing_dataset, adversarial_epochs,
                          net_name="{} with PGD adversarial training".format(net_name), train_attack=attacks_new.PGD,
                          attack_training_hps_gen=pgd_training_hps_gen,
                          load_checkpoint=load_checkpoint, save_checkpoint=save_checkpoint, show_plots=show_plots,
                          save_plots=save_plots,
                          show_validation_accuracy_each_epoch=show_validation_accuracy_each_epoch)

    if test_pgdl2:
        adversarial_epochs.restart()
        net.apply(helper_plots.weight_reset)
        pgdl2_robust_net = net.to(device)
        experiment_1_func(pgdl2_robust_net, _loss_fn, _training_dataset, _testing_dataset, adversarial_epochs,
                          net_name="{} with PGDl2 adversarial training".format(net_name), train_attack=attacks_new.PGDl2,
                          attack_training_hps_gen=pgdl2_training_hps_gen,
                          load_checkpoint=load_checkpoint, save_checkpoint=save_checkpoint, show_plots=show_plots,
                          save_plots=save_plots,
                          show_validation_accuracy_each_epoch=show_validation_accuracy_each_epoch)

    if test_tpgd:
        adversarial_epochs.restart()
        net.apply(helper_plots.weight_reset)
        tpgd_robust_net = net.to(device)
        experiment_1_func(tpgd_robust_net, _loss_fn, _training_dataset, _testing_dataset, adversarial_epochs,
                          net_name="{} with TPGD adversarial training".format(net_name), train_attack=attacks_new.TPGD,
                          attack_training_hps_gen=tpgd_training_hps_gen,
                          load_checkpoint=load_checkpoint, save_checkpoint=save_checkpoint, show_plots=show_plots,
                          save_plots=save_plots,
                          show_validation_accuracy_each_epoch=show_validation_accuracy_each_epoch)    


# Experiment 2+3: Build robust networks + Compare adversarial trainings
if __name__ == '__main__' and run_experiment_2:
    exp2_net = network_architecture().to(device)

    net_name = exp2_net.name
    logger.new_section()
    logger.log_print("Experiment 2 on {}".format(net_name))
    experiment_2_func(exp2_net, _loss_fn, _training_dataset, _testing_dataset, adv_epochs,
                      net_name=net_name,
                      save_checkpoint=configs_lenet5.save_checkpoints,
                      load_checkpoint=configs_lenet5.load_checkpoints,
                      show_plots=configs_lenet5.show_attacks_plots,
                      save_plots=configs_lenet5.save_attacks_plots,
                      show_validation_accuracy_each_epoch=configs_lenet5.show_validation_accuracy_each_epoch)

# Experiment 4: Capacity and robustness
if __name__ == '__main__' and run_experiment_3:
    logger.log_print("Experiment 4:")
    inc_capacity_nets = []
    base_net_params = {
        "extras_blocks_components": [],
        # "p_dropout": 0.1,
        "activation": torch.nn.ReLU,
    }
    if dataset_name == "traffic_signs":
        base_net_params["out_size"] = 43
        base_net_params["in_wh"] = 32
        in_channels = 3

    for i in range(1,7):
        if  1<= i <= 7:
            base_net_params["channels_lst"] = [in_channels, 2 ** (i - 1), 2 ** (i + 1)]
            base_net_params["#FC_Layers"] = 1
            base_net_params["CNN_out_channels"] = 10 * i
            
        cap_net = models_lenet5.create_conv_nn(base_net_params)
        inc_capacity_nets.append(cap_net)
        
    #pytorch_total_params = "parameters_{}".format(sum(p.numel() for p in net.parameters()))

    # run experiment 1 and 2 on each capacity
    for i, net in enumerate(inc_capacity_nets):
        net = net.to(device)
        net_name = "capacity_{}".format(i + 1)
        epochs.restart()
        experiment_1_func(net, _loss_fn, _training_dataset, _testing_dataset, epochs, net_name=net_name)
        adv_epochs.restart()
        experiment_2_func(net, _loss_fn, _training_dataset, _testing_dataset, adv_epochs, net_name=net_name)
