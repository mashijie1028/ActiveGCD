import math
from scipy.optimize import linear_sum_assignment as linear_assignment
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from model import DINOHead, info_nce_logits, SupConLoss, DistillLoss, ContrastiveLearningViewGenerator, get_params_groups
from utils_simgcd.general_utils import AverageMeter, init_experiment
from utils_simgcd.cluster_and_log_utils_al import log_accs_from_preds   # NOTE!!! AL utils, return ind_map


def get_labeled_mapping(model, train_labeled_loader_mapping, al_labeled_loader_mapping, args):
    model.eval()

    preds_ind_mapping, targets_ind_mapping = [], []
    for batch_idx, (images, label, _) in enumerate(tqdm(train_labeled_loader_mapping)):
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            _, _, logits = model(images)
            preds_ind_mapping.append(logits.argmax(1).cpu().numpy())
            targets_ind_mapping.append(label.cpu().numpy())

    for batch_idx, (images, label, _) in enumerate(tqdm(al_labeled_loader_mapping)):
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            _, _, logits = model(images)
            preds_ind_mapping.append(logits.argmax(1).cpu().numpy())
            targets_ind_mapping.append(label.cpu().numpy())

    preds_ind_mapping = np.concatenate(preds_ind_mapping)
    targets_ind_mapping = np.concatenate(targets_ind_mapping)

    D = args.num_labeled_classes + args.num_unlabeled_classes   # NOTE!!! NO + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(preds_ind_mapping.size):
        w[preds_ind_mapping[i], targets_ind_mapping[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    ind_map = {j: i for i, j in ind}

    return ind_map



def map_labels(original_labels, ind_map):
    mapped_labels = torch.tensor([ind_map[int(i)] for i in original_labels], dtype=original_labels.dtype, device=original_labels.device)
    return mapped_labels


def update_ema_variables(model, ema_model, alpha):
    # Use the true average until the exponential average is more correct
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)



def plot_same_mapping_pair(ind_map_train, ind_map_test, ind_map_labeled, args, log_str='Before AL: model'):
    same_mapping_pair = 0
    for i in range(len(ind_map_train)):
        if ind_map_train[i] == ind_map_test[i]:
            same_mapping_pair += 1
    same_ratio = same_mapping_pair / len(ind_map_train)
    #args.logger.info('(Before AL: model) Same ind_map on (unlabeled) train and test data: [{}]    Same mapping pair ratio: [{}]'.format(ind_map_train == ind_map_test, same_ratio))
    args.logger.info('({}) Same mapping pair ratio on train & test: [{}]'.format(log_str, same_ratio))

    same_mapping_pair = 0
    for i in range(len(ind_map_train)):
        if ind_map_train[i] == ind_map_labeled[i]:
            same_mapping_pair += 1
    same_ratio = same_mapping_pair / len(ind_map_train)
    args.logger.info('({}) Same mapping pair ratio on train & labeled: [{}]'.format(log_str, same_ratio))

    same_mapping_pair = 0
    for i in range(len(ind_map_test)):
        if ind_map_test[i] == ind_map_labeled[i]:
            same_mapping_pair += 1
    same_ratio = same_mapping_pair / len(ind_map_test)
    args.logger.info('({}) Same mapping pair ratio on test & labeled: [{}]'.format(log_str, same_ratio))



# train ema
########################################################################################################################
########################################################################################################################
def train_al_ema(student, student_ema, train_loader, al_labeled_loader, test_loader, unlabelled_train_loader,
                 train_labeled_loader_ind_mapping, al_labeled_loader_ind_mapping, args, current_round):
    params_groups = get_params_groups(student)
    optimizer = SGD(params_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 1e-3,
        )


    cluster_criterion = DistillLoss(
                        args.warmup_teacher_temp_epochs,
                        args.epochs,
                        args.n_views,
                        args.warmup_teacher_temp,
                        args.teacher_temp,
                    )

    # inductive
    best_test_acc_lab = 0
    best_test_acc_ubl = 0
    best_test_acc_all = 0
    # transductive
    best_train_acc_lab = 0
    best_train_acc_ubl = 0
    best_train_acc_all = 0

    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        loss_record_al = AverageMeter()

        student.train()
        #student_ema.train()
        args.logger.info('='*120)
        args.logger.info('='*120)
        args.logger.info('Begin Epoch {} (Round {})'.format(epoch, current_round))

        '''1. train on original mixed train_loader'''
        args.logger.info('Train on original training data (before AL)')
        args.logger.info('='*100)
        for batch_idx, batch in enumerate(train_loader):
            images, class_labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]

            class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()
            images = torch.cat(images, dim=0).cuda(non_blocking=True)

            _, student_proj, student_out = student(images)
            teacher_out = student_out.detach()

            # clustering, sup
            sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
            sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
            cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)
            # print(len(class_labels[mask_lab]), len(class_labels[~mask_lab]))   # e.g. 61 67/59 69/72 56/64 64

            # clustering, unsup
            cluster_loss = cluster_criterion(student_out, teacher_out, epoch)
            avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
            me_max_loss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
            cluster_loss += args.memax_weight * me_max_loss

            # represent learning, unsup
            contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj)
            contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

            # representation learning, sup
            student_proj = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
            student_proj = torch.nn.functional.normalize(student_proj, dim=-1)
            sup_con_labels = class_labels[mask_lab]
            sup_con_loss = SupConLoss()(student_proj, labels=sup_con_labels)

            pstr = ''
            pstr += f'cls_loss: {cls_loss.item():.4f} '
            pstr += f'cluster_loss: {cluster_loss.item():.4f} '
            pstr += f'sup_con_loss: {sup_con_loss.item():.4f} '
            pstr += f'contrastive_loss: {contrastive_loss.item():.4f} '

            loss = 0
            loss += (1 - args.sup_weight) * cluster_loss + args.sup_weight * cls_loss
            loss += (1 - args.sup_weight) * contrastive_loss + args.sup_weight * sup_con_loss

            # Loss
            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            # ema
            ################################################################################################################
            #args.logger.info('perform EMA averaging...')
            update_ema_variables(student, student_ema, args.ema_decay)
            ################################################################################################################

            if batch_idx % args.print_freq == 0:
                args.logger.info('Epoch: [{}][{}/{}]    loss: {:.5f}    {}'
                            .format(epoch, batch_idx, len(train_loader), loss.item(), pstr))


        args.logger.info('(Before AL) Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record.avg))

        # evaluate on model
        ################################################################################################################
        all_acc, old_acc, new_acc, ind_map_train = test_al(student, unlabelled_train_loader, epoch=epoch, save_name='Train ACC Unlabelled', args=args)
        args.logger.info('(Before AL: model) Testing on disjoint test set...')
        all_acc_test, old_acc_test, new_acc_test, ind_map_test = test_al(student, test_loader, epoch=epoch, save_name='Test ACC', args=args)

        # ind_map_labeled
        ind_map_labeled = get_labeled_mapping(student, train_labeled_loader_ind_mapping, al_labeled_loader_ind_mapping, args)

        args.logger.info('(Before AL: model) Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
        args.logger.info('(Before AL: model) Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test, old_acc_test, new_acc_test))

        #plot_same_mapping_pair(ind_map_train, ind_map_test, ind_map_labeled, args, log_str='Before AL: model')


        # evaluate on model ema
        ################################################################################################################
        args.logger.info('(Before AL: model ema) Testing on unlabelled examples in the training data...')
        all_acc_ema, old_acc_ema, new_acc_ema, ind_map_train_ema = test_al(student_ema, unlabelled_train_loader, epoch=epoch, save_name='Train ACC Unlabelled', args=args)
        args.logger.info('(Before AL: model ema) Testing on disjoint test set...')
        all_acc_test_ema, old_acc_test_ema, new_acc_test_ema, ind_map_test_ema = test_al(student_ema, test_loader, epoch=epoch, save_name='Test ACC', args=args)

        # ind_map_labeled
        ind_map_labeled_ema = get_labeled_mapping(student_ema, train_labeled_loader_ind_mapping, al_labeled_loader_ind_mapping, args)

        args.logger.info('(Before AL: model ema) Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_ema, old_acc_ema, new_acc_ema))
        args.logger.info('(Before AL: model ema) Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test_ema, old_acc_test_ema, new_acc_test_ema))

        #plot_same_mapping_pair(ind_map_train_ema, ind_map_test_ema, ind_map_labeled_ema, args, log_str='Before AL: model ema')


        # record init ind_map_test
        if epoch == 0:
            ind_map_init = ind_map_labeled_ema


        '''2. train on AL selected labeled data'''
        args.logger.info('Train on selected labeled data (AL)')
        args.logger.info('='*100)
        for batch_idx, batch in enumerate(al_labeled_loader):
            images, class_labels, uq_idxs = batch   # NOTE!!! No mask.
            class_labels = class_labels.cuda(non_blocking=True)
            class_labels_mapped = map_labels(class_labels, ind_map_labeled_ema)   # NOTE!!! map the labels, ind_map ema NOT!!! ind_map_test
            images = torch.cat(images, dim=0).cuda(non_blocking=True)
            _, student_proj, student_out = student(images)

            # clustering, sup
            sup_logits = torch.cat([f for f in (student_out / 0.1).chunk(2)], dim=0)
            # sup_labels = torch.cat([class_labels for _ in range(2)], dim=0)
            sup_labels_mapped = torch.cat([class_labels_mapped for _ in range(2)], dim=0)   # NOTE!!! mapped labels, new class might emerge
            cls_loss_al = nn.CrossEntropyLoss()(sup_logits, sup_labels_mapped)

            # representation learning, sup
            student_proj = torch.cat([f.unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
            student_proj = torch.nn.functional.normalize(student_proj, dim=-1)
            sup_con_labels = class_labels
            sup_con_loss_al = SupConLoss()(student_proj, labels=sup_con_labels)

            pstr = ''
            pstr += f'cls_loss_al: {cls_loss_al.item():.4f} '
            pstr += f'sup_con_loss_al: {sup_con_loss_al.item():.4f} '

            loss_al = 0
            # loss_al += args.al_weight * args.sup_weight * cls_loss_al
            # loss_al += args.al_weight * args.sup_weight * sup_con_loss_al
            loss_al += args.al_weight * args.al_cls_weight * args.sup_weight * cls_loss_al
            loss_al += args.al_weight * args.al_supcon_weight * args.sup_weight * sup_con_loss_al

            # Loss
            loss_record_al.update(loss_al.item(), class_labels.size(0))
            optimizer.zero_grad()

            loss_al.backward()
            optimizer.step()

            # ema
            ################################################################################################################
            #args.logger.info('perform EMA averaging...')
            update_ema_variables(student, student_ema, args.ema_decay)
            ################################################################################################################

            if batch_idx % args.print_freq_al == 0:
                args.logger.info('Epoch: [{}][{}/{}]    loss: {:.5f}    {}'
                            .format(epoch, batch_idx, len(al_labeled_loader), loss_al.item(), pstr))

        args.logger.info('(After AL) Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record_al.avg))

        # evaluate on model
        ################################################################################################################
        args.logger.info('(After AL: model) Testing on unlabelled examples in the training data...')
        all_acc, old_acc, new_acc, ind_map_train = test_al(student, unlabelled_train_loader, epoch=epoch, save_name='Train ACC Unlabelled', args=args)
        args.logger.info('(After AL: model) Testing on disjoint test set...')
        all_acc_test, old_acc_test, new_acc_test, ind_map_test = test_al(student, test_loader, epoch=epoch, save_name='Test ACC', args=args)

        # ind_map_labeled
        ind_map_labeled = get_labeled_mapping(student, train_labeled_loader_ind_mapping, al_labeled_loader_ind_mapping, args)

        args.logger.info('(After AL: model) Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
        args.logger.info('(After AL: model) Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test, old_acc_test, new_acc_test))

        #plot_same_mapping_pair(ind_map_train, ind_map_test, ind_map_labeled, args, log_str='After AL: model')

        # Step schedule
        exp_lr_scheduler.step()

        save_dict = {
            'model': student.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
        }

        #torch.save(save_dict, args.model_path)
        #args.logger.info("model saved to {}.".format(args.model_path))
        torch.save(save_dict, args.model_path[:-3] + f'_round{current_round}.pt')
        args.logger.info("model saved to {}.".format(args.model_path[:-3] + f'_round{current_round}.pt'))

        # check for best test acc (After AL)
        if all_acc_test > best_test_acc_all:

            args.logger.info('(After AL: model) Best Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test, old_acc_test, new_acc_test))

            # inductive
            best_test_acc_lab = old_acc_test
            best_test_acc_ubl = new_acc_test
            best_test_acc_all = all_acc_test

            # transductive
            # best_train_acc_lab = old_acc
            # best_train_acc_ubl = new_acc
            # best_train_acc_all = all_acc
        ################################################################################################################



        # evaluate on model ema
        ################################################################################################################
        args.logger.info('(After AL: model ema) Testing on unlabelled examples in the training data...')
        all_acc_ema, old_acc_ema, new_acc_ema, ind_map_train_ema = test_al(student_ema, unlabelled_train_loader, epoch=epoch, save_name='Train ACC Unlabelled', args=args)
        args.logger.info('(After AL: model ema) Testing on disjoint test set...')
        all_acc_test_ema, old_acc_test_ema, new_acc_test_ema, ind_map_test_ema = test_al(student_ema, test_loader, epoch=epoch, save_name='Test ACC', args=args)

        # ind_map_labeled
        ind_map_labeled_ema = get_labeled_mapping(student_ema, train_labeled_loader_ind_mapping, al_labeled_loader_ind_mapping, args)

        args.logger.info('(After AL: model ema) Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_ema, old_acc_ema, new_acc_ema))
        args.logger.info('(After AL: model ema) Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test_ema, old_acc_test_ema, new_acc_test_ema))

        #plot_same_mapping_pair(ind_map_train_ema, ind_map_test_ema, ind_map_labeled_ema, args, log_str='After AL: model ema')

        # Step schedule
        exp_lr_scheduler.step()

        save_dict = {
            'model': student_ema.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
        }

        #torch.save(save_dict, args.model_path)
        #args.logger.info("model saved to {}.".format(args.model_path))
        torch.save(save_dict, args.model_path[:-3] + f'_ema_round{current_round}.pt')
        args.logger.info("model saved to {}.".format(args.model_path[:-3] + f'_ema_round{current_round}.pt'))

        # check for best test acc (After AL)
        if all_acc_test_ema > best_test_acc_all:

            args.logger.info('(After AL: model ema) Best Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test_ema, old_acc_test_ema, new_acc_test_ema))

            # inductive
            best_test_acc_lab = old_acc_test_ema
            best_test_acc_ubl = new_acc_test_ema
            best_test_acc_all = all_acc_test_ema

            # transductive
            # best_train_acc_lab = old_acc
            # best_train_acc_ubl = new_acc
            # best_train_acc_all = all_acc
        ################################################################################################################
        ################################################################################################################
        # record final ind_map
        ind_map_final = ind_map_labeled_ema

        args.logger.info(f'Exp Name: {args.exp_name}')
        args.logger.info(f'Metrics with best model on test set: All: {best_test_acc_all:.4f} Old: {best_test_acc_lab:.4f} New: {best_test_acc_ubl:.4f}')


    # ind_map stability at the beginning and end of this round
    same_mapping_pair_round = 0
    for i in range(len(ind_map_final)):
        if ind_map_final[i] == ind_map_init[i]:
            same_mapping_pair_round += 1
    same_ratio_round = same_mapping_pair_round / len(ind_map_final)
    args.logger.info('ind_map_labeled_ema_init: ' + str(ind_map_init))
    args.logger.info('ind_map_labeled_ema_final: ' + str(ind_map_final))
    args.logger.info('Round {}: Same ind_map of this : [{}], Same mapping pair ratio: [{}]'.format(current_round, ind_map_init == ind_map_final, same_ratio_round))

    # best inductive results
    return best_test_acc_all, best_test_acc_lab, best_test_acc_ubl


# test
########################################################################################################################
########################################################################################################################
def test_al(model, test_loader, epoch, save_name, args):

    model.eval()

    preds, targets = [], []
    mask = np.array([])
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            _, _, logits = model(images)
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    all_acc, old_acc, new_acc, ind_map = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)

    return all_acc, old_acc, new_acc, ind_map
