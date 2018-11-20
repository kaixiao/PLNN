#!/usr/bin/env python
import argparse
import json
import os
import time
import torch
import torchvision

from plnn.mip_solver import MIPNetwork
from plnn.modules import View
from plnn.model import load_mat_network
from torch.autograd import Variable
from torch import nn

def main():
    parser = argparse.ArgumentParser(description="Read a .mat file"
                                     "and prove robustness over the dataset.")

    parser.add_argument('mat_infile', type=str,
                        help='.mat file to prove.')
    parser.add_argument('adv_perturb', type=float,
                        help='What proportion to use.')
    parser.add_argument('--sym_bounds', action='store_true')
    parser.add_argument('--use_obj_function', action='store_true')
    parser.add_argument('--bound_type', type=str)
    parser.add_argument('--modulo', type=int, default=1,
                        help="Use this to specify which part of the samples to run in this process."
                        "This process will only run those for which idx %% modulo == modulo_arg ")
    parser.add_argument('--modulo_arg', type=int, default=0)
    parser.add_argument('--ids_to_run', type=str,
                        help='List of ids to run the verification on')
    parser.add_argument('--dataset', type=str,
                        help='What dataset to run the verification on',
                        default='mnist', choices=['mnist', 'cifar10'])
    parser.add_argument('--result_folder', type=str,
                        help='Where to store the results of the verification')
    parser.add_argument('--dump_images', action='store_true')
    args = parser.parse_args()

    layers = load_mat_network(args.mat_infile)

    if args.result_folder is None:
        verif_result_folder = "weights/{}_verif_result".format(args.dataset)
    else:
        verif_result_folder = args.result_folder
    if not os.path.exists(verif_result_folder):
        os.makedirs(verif_result_folder)

    if args.dataset == 'mnist':
        mnist_data = 'weights/mnistData/'
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: x.view(-1))
        ])
        test_dataset = torchvision.datasets.MNIST(mnist_data,
                                                  train=False,
                                                  download=True,
                                                  transform=transform)
    elif args.dataset == 'cifar10':
        cifar_data = 'weights/cifar10Data/'

        center_crop = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop((28, 28)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: x.permute(1,2,0).contiguous().view(-1))
        ])
        test_dataset = torchvision.datasets.CIFAR10(cifar_data,
                                                    train=False,
                                                    download=True,
                                                    transform=center_crop)

    test_loader_256batch = torch.utils.data.DataLoader(test_dataset,
                                                       batch_size=256,
                                                       shuffle=False)


    test_net = nn.Sequential(*layers)
    test_net.eval()
    correct = 0
    for sample_idx, (data, target) in enumerate(test_loader_256batch):
        var_data = Variable(data, volatile=True)
        output = test_net(var_data)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum()
    correct = correct.item()
    accuracy = 100. * correct / len(test_dataset)
    print("Nominal accuracy on test set: {0:.2f} %".format(accuracy))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1, shuffle=False
    )
    if args.ids_to_run is not None:
        with open(args.ids_to_run, 'r') as ids_list_file:
            txt = ids_list_file.read()
        str_ids_list = txt.split()
        ids_list = list(map(int, str_ids_list))

    else:
        ids_list = range(len(test_dataset))
    to_process = [idx for cnt, idx in enumerate(ids_list) if cnt % args.modulo == args.modulo_arg]

    # #### DEBUG
    # # with open(verif_result_folder + '/incorrect_ids.txt', 'r') as inc_ids:
    # #     all_inc_ids = []
    # #     for inc_id in inc_ids.readlines():
    # #         all_inc_ids.append(int(inc_id))
    # all_inc_ids = [951]

    # for inc_id in all_inc_ids:
    #     ex_idx = inc_id
    #     print(f"Dealing with example {ex_idx}")
    #     inp = mnist_test.test_data[ex_idx]
    #     label = mnist_test.test_labels[ex_idx]
    #     var_data = Variable(inp.float().view(-1, 28*28)/255.0, volatile=True)
    #     output = test_net(var_data)
    #     _, pred = output.data.max(1, keepdim=True)
    #     print(f"Output is {output}")
    #     print(f"Prediction is {pred[0][0]}")
    #     print(f"Target is {label}")


    #     net_layers = layers
    #     data = mnist_test.test_data[ex_idx]
    #     data_lb = data.float().view(28*28)/255.0 - args.adv_perturb
    #     data_ub = data.float().view(28*28)/255.0 + args.adv_perturb
    #     domain = torch.clamp(torch.stack((data_lb, data_ub), dim=1),
    #                              min=0, max=1)

    #     neg_layer = nn.Linear(1, 1)
    #     neg_layer.weight.data.fill_(-1)
    #     neg_layer.bias.data.fill_(0)

    #     additional_lin_layer = nn.Linear(10, 9, bias=True)
    #     lin_weights = additional_lin_layer.weight.data
    #     lin_weights.fill_(0)
    #     lin_bias = additional_lin_layer.bias.data
    #     lin_bias.fill_(0)
    #     to = 0
    #     gt = label
    #     for cls in range(10):
    #         if cls != gt:
    #             lin_weights[to, cls] = 1
    #             lin_weights[to, gt] = -1
    #             to += 1

    #     verif_layers = layers + [additional_lin_layer,
    #                              View((1, 9)),
    #                              nn.MaxPool1d(9),
    #                              View((1,)),
    #                              neg_layer]

    #     artificial_net = nn.Sequential(*verif_layers)
    #     artificial_net.eval()
    #     art_output = artificial_net(var_data)
    #     print(f"\nTo prove net output is {art_output}")
    #     mip_network = MIPNetwork(verif_layers)
    #     mip_network.setup_model(domain,
    #                             sym_bounds=args.sym_bounds,
    #                             use_obj_function=args.use_obj_function,
    #                             interval_analysis=args.interval_analysis)

    #     sat, solution, nb_visited_states = mip_network.solve(domain)
    #     import IPython; IPython.embed();
    #     import sys; sys.exit()
    #     print(f"Counterexample search for example {ex_idx}: satResult is {sat}\n\n")
    # #### END DEBUG

    neg_layer = nn.Linear(1, 1)
    neg_layer.weight.data.fill_(-1)
    neg_layer.bias.data.fill_(0)
    total_time = 0
    total_build_time = 0
    total_solve_time = 0
    total_solves = 0
    total_robust = 0
    total_nonrobust = 0
    total_timeout = 0
    for sp_idx, (data, target) in enumerate(test_loader):
        if sp_idx not in to_process:
            continue

        example_res_file = verif_result_folder + "/{}_sample.txt".format(sp_idx)
        if os.path.isfile(example_res_file):
            continue


        print("{} \tExample {} starting".format(time.ctime(), sp_idx))
        build_start = time.time()
        net_layers = layers

        data_lb = data - args.adv_perturb
        data_ub = data + args.adv_perturb
        domain = torch.clamp(torch.cat([data_lb, data_ub], dim=0).transpose(1,0),
                             min=0, max=1)

        additional_lin_layer = nn.Linear(10, 9, bias=True)
        lin_weights = additional_lin_layer.weight.data
        lin_weights.fill_(0)
        lin_bias = additional_lin_layer.bias.data
        lin_bias.fill_(0)
        to = 0
        gt = target[0]
        for cls in range(10):
            if cls != gt:
                lin_weights[to, cls] = 1
                lin_weights[to, gt] = -1
                to += 1

        verif_layers = layers + [additional_lin_layer,
                                 View((1, 9)),
                                 nn.MaxPool1d(9),
                                 View((1,)),
                                 neg_layer]


        print("{} \tExample {} has spec.".format(time.ctime(), sp_idx))
        mip_network = MIPNetwork(verif_layers)
        mip_network.setup_model(domain,
                                sym_bounds=args.sym_bounds,
                                use_obj_function=args.use_obj_function,
                                bounds=args.bound_type)
        build_end = time.time()
        build_time = build_end-build_start

        print("{} \tExample {} has MIP setup in {} seconds.".format(time.ctime(), sp_idx, build_time))
        solve_start = time.time()
        sat, solution, nb_visited_states = mip_network.solve(domain, timeout=10)
        solve_end = time.time()
        solve_time = solve_end-solve_start
        print("{} \tExample {} has solve in {} seconds.".format(time.ctime(), sp_idx, solve_time))
        solve_and_build = build_time + solve_time
        total_time += solve_and_build
        total_build_time += build_time
        total_solve_time += solve_time
        total_solves += 1

        if sat is False:	
            total_robust += 1
            print("{} \tExample {} is Robust.".format(time.ctime(), sp_idx))
            with open(example_res_file, 'w') as res_file:
                res_file.write('Robust\n')
                res_file.write('{}\n'.format(build_time))
                res_file.write('{}\n'.format(solve_time))
        elif sat is True:
            total_nonrobust += 1
            print("{} \tExample {} is not Robust.".format(time.ctime(), sp_idx))
            # adv_example = Variable(solution[0].view(1, -1), volatile=True)
            # pred_on_adv = test_net(adv_example)
            # print("{} \tPredictions: {}".format(time.ctime(), pred_on_adv.data))
            # print("{} \tGT is: {}".format(time.ctime(), target))
            with open(example_res_file, 'w', encoding='utf-8') as res_file:
                res_file.write('NonRobust\n')
                res_file.write('{}\n'.format(build_time))
                res_file.write('{}\n'.format(solve_time))
                # sol_str = 'Input: {}\n'.format(solution[0])
                # res_file.write(sol_str)
                # res_file.write('Pred on adv: {}\n'.format(pred_on_adv.data))
                # gt_str = 'GT is : {}\n'.format(target[0])
                # res_file.write(gt_str)


            if args.dump_images is True:
                # Map the plot between the nominal sample and the adversarial
                var_data = Variable(data, volatile=True)
                nom_to_adv = adv_example - var_data
                steps = Variable(torch.arange(0, 1, step=1.0/1000).view(-1, 1),volatile=True)
                all_on_path = var_data + torch.matmul(steps, nom_to_adv)

                all_preds_on_path = test_net(all_on_path)
                py_all_preds = all_preds_on_path.data.numpy().tolist()
                preds_path = verif_result_folder + "/{}_vals.json".format(sp_idx)
                with open(preds_path, 'w') as preds_file:
                  json.dump(py_all_preds, preds_file)

                # Generate the images to save
                orig_data = data.view(28, 28)
                adv_data = solution[0].view(28, 28)

                orig_path = verif_result_folder + "/{}_original.png".format(sp_idx)
                adv_path = verif_result_folder + "/{}_adversarial.png".format(sp_idx)
                torchvision.utils.save_image(orig_data, orig_path)
                torchvision.utils.save_image(adv_data, adv_path)
        elif sat is None:
            total_timeout += 1
            print("{} \t Example {} failure.".format(time.ctime(), sp_idx))
            with open(example_res_file, 'w') as res_file:
                res_file.write('Verification Failure\n')
                res_file.write('{}\n'.format(build_time))
                res_file.write('{}\n'.format(solve_time))

        print("")

    final_res_file = verif_result_folder + "/summary_mod{}_arg{}.txt".format(args.modulo, args.modulo_arg)
    print("Average Build Time Summary: {}".format(total_build_time/total_solves))
    print("Average Solve Time Summary: {}".format(total_solve_time/total_solves))
    print("Average Solve + Build Time Summary: {}".format(total_time/total_solves))
    print("Robust: {}, NonRobust: {}, Timeout: {}".format(total_robust, total_nonrobust, total_timeout))
    with open(final_res_file, 'w') as res_file:
        res_file.write('AvgBuild,{}\n'.format(total_build_time/total_solves))
        res_file.write('AvgSolve,{}\n'.format(total_solve_time/total_solves))
        res_file.write('AvgTotal,{}\n'.format(total_time/total_solves))
        res_file.write('Robust,{}\n'.format(total_robust))
        res_file.write('NonRobust,{}\n'.format(total_nonrobust))
        res_file.write('Timeout,{}\n'.format(total_timeout))

if __name__ == '__main__':
    main()
