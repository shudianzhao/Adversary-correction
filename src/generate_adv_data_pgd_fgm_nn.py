import torch
import numpy as np
import torch.nn as nn
from DNN_train_save_model import Net
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)
from easydict import EasyDict


def sort_adver_data(model: nn.Module, data: dict, target_pred:dict):
    # return: sorted adversarial data with regard to x_j-x_i
    data_dict_sort = {}
    data_score_sort_dict = {}
    target_score_sort_dict = {}

    for label in range(10):
        if isinstance(model, Net):
            data_score = [- model(torch.Tensor(input).view(-1, 28*28)).detach().numpy().max() +
                          model(torch.Tensor(input).view(-1, 28*28)).detach().numpy()[0, label]
                          for input in data[label]]
        else:
            data_score = [- model(torch.Tensor(input).view(-1,
                                                           input.shape[0],
                                                           input.shape[1],
                                                           input.shape[2])).detach().numpy().max() +
                          model(torch.Tensor(input).view(-1,
                                                         input.shape[0],
                                                         input.shape[1],
                                                         input.shape[2])).detach().numpy()[0, label]
                          for input in data[label]]
    
        data_dict_sort[label] = data[label][np.array(data_score).argsort()]         
        data_score_sort_dict[label] = np.array(data_score)[np.array(data_score).argsort()]
        target_score_sort_dict[label] = target_pred[label][np.array(data_score).argsort()]

    return data_dict_sort, data_score_sort_dict, target_score_sort_dict






def generate_adversarial_numControl(model: nn.Module,
                                    test_loader,
                                    sort: bool = False,
                                    radius=0.1,
                                    pgd_iter=40,
                                    pgd_stepsize=0.01,
                                    type=['pgd']):

    # return adversarial data by pgd/fgsm with a descending order and balanced with labels
    model.eval()
    data_pgd_total = None
    data_fgm_total = None
    target_total = None
    target_pred_pgd_total = None

    for data, target in test_loader:
        if isinstance(model, Net):
            data = data.view(-1, 28*28)
        data_fgm = fast_gradient_method(model, data, radius, np.inf)
        data_pgd = projected_gradient_descent(model, data, radius, pgd_stepsize, pgd_iter, np.inf)
        target_pred_pgd = model(data_pgd).argmax(dim=1, keepdim=True)  # model prediction on PGD adversarial examples
        
        if data_pgd_total is None:
            data_pgd_total = data_pgd.detach().numpy().copy()
            if 'fgm' in type:
                data_fgm_total = data_fgm.detach().numpy().copy()
            target_total = target.detach().numpy().copy()

            target_pred_pgd_total = target_pred_pgd.detach().numpy().copy()[:, 0]
        else:
            data_pgd_total = np.append(data_pgd_total, data_pgd.detach().numpy(), axis=0)
            target_pred_pgd_total = np.append(target_pred_pgd_total, target_pred_pgd.detach().numpy()[:, 0], axis=0)

            if 'fgm' in type:
                data_fgm_total = np.append(data_fgm_total, data_fgm.detach().numpy(), axis=0)
            target_total = np.append(target_total, target.detach().numpy(), axis=0)
               
        # separate data with labels
        data_fgm_total_idx = {}
        data_pgd_total_idx = {}
        target_pgd_pred_total_idx = {}
        target_idx = {}
        for idx in range(10):
            if 'fgm' in type:
                data_fgm_total_idx[idx] = data_fgm_total[target_total == idx, :]
            
            data_pgd_total_idx[idx] = data_pgd_total[target_total == idx, :]
            target_idx[idx] = target_total[target_total == idx]
            target_pgd_pred_total_idx[idx] = target_pred_pgd_total[target_total == idx]
            # pass
    
   

    if not sort:
        return data_fgm_total_idx, data_pgd_total_idx, {}
    else:
        data_fgm_total_idx_score_desc = {}
        data_pgd_total_idx_score_desc = {}
        if 'fgm' in type:
            data_fgm_total_idx_score_desc, data_fgm_score_desc, _ = sort_adver_data(model, data_fgm_total_idx)
        else:
            data_fgm_score_desc = {}
        data_pgd_total_idx_score_desc, data_pgd_score_desc, target_pgd_total_idx_score_desc  = sort_adver_data(model, data_pgd_total_idx, target_pgd_pred_total_idx)

    return (data_fgm_total_idx_score_desc,
            data_fgm_score_desc,
            data_pgd_total_idx_score_desc,
            data_pgd_score_desc,
            target_pgd_total_idx_score_desc)



