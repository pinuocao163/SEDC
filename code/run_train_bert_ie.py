import os

import argparse
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import IEMOCAPDataset, IEMOCAPRobertaCometDataset
from model import DialogueCRN
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from loss import FocalLoss
import torch.nn.functional as F

import visdom
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

torch.autograd.set_detect_anomaly(True)

import warnings
warnings.filterwarnings("ignore")


def seed_everything(seed=2021):
    print(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid * size)
    np.random.shuffle(idx)  # shuffle for training data
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_IEMOCAP_bert_loaders(path=None, batch_size=32, num_workers=0, pin_memory=False, valid_rate=0.1):
    trainset = IEMOCAPRobertaCometDataset(path=path, split='train-valid')
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid_rate)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPRobertaCometDataset(path=path, split='test')
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits_simple, contrast_label):
        loss_simple = self.criterion(logits_simple, contrast_label)

        return loss_simple

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=5),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(1024, 256),
            nn.PReLU(),

            nn.Linear(256, 256),
            nn.PReLU(),

            nn.Linear(256, 2))

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward_once(self, x):
        # output = self.cnn1(x)
        # output = output.view(output.size()[0], -1)
        output = self.fc1(x)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

class bianhuan(nn.Module):
    def __init__(self):
        super(bianhuan, self).__init__()
        self.fc = nn.Linear(1024, 3)
        self.relu = nn.PReLU()

    def forward(self, input1, input2, input3):
        output1 = self.fc(input1)
        output1 = self.relu(output1)
        # output2 = torch.nn.functional.one_hot(input2)
        # output3 = torch.nn.functional.one_hot(input3)
        # output2 = output2.float()
        # output3 = output3.float()

        # 替换值
        replacement_values = {
            0: [0.960, 0.732, 0.850],
            1: [0.052, 0.288, 0.164],
            2: [0.469, 0.184, 0.357],
            3: [0.167, 0.865, 0.657],
            4: [0.908, 0.931, 0.709],
            5: [0.060, 0.730, 0.280]
        }
        # 创建结果列表
        result_1 = []
        result_2 = []
        # 替换张量中的值
        for value in input2:
            result_1.extend(replacement_values[int(value)])
        for value in input3:
            result_2.extend(replacement_values[int(value)])

        # 将结果列表转换为张量
        result_tensor_1 = torch.tensor(result_1)
        result_tensor_2 = torch.tensor(result_2)

        # 重塑为合适的格式
        result_tensor_1 = result_tensor_1.reshape(-1, 3).cuda()
        result_tensor_2 = result_tensor_2.reshape(-1, 3).cuda()

        return output1, result_tensor_1, result_tensor_2

def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.show()


mnist_classes = ['0', '1', '2', '3', '4', '5']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
          '#9467bd', '#8c564b']

def plot_embeddings_3d(embeddings, targets):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    embeddings = embeddings.cpu().detach().numpy()
    targets = targets.cpu().numpy()
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    labels = ['hap', 'sad', 'neu', 'ang', 'exc', 'fru']

    for i in range(6):
        inds = np.where(targets == i)[0]
        ax.scatter(embeddings[inds, 0], embeddings[inds, 1], embeddings[inds, 2], alpha=1, color=colors[i], label=labels[i])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    # plt.show()


def plot_embeddings(embeddings, targets, xlim=None, ylim=None):
    plt.figure(figsize=(10, 10))
    embeddings = embeddings.cpu().detach().numpy()
    targets = targets.cpu().numpy()
    # for i in range(6):
    #     inds = np.where(targets == i)[0]
    #     plt.scatter(embeddings[inds, 0], embeddings[inds, 1], alpha=1, color=colors[i])
    inds = np.where(targets == 0)[0]
    plt.scatter(embeddings[inds, 0], embeddings[inds, 1], alpha=1, color=colors[0])
    inds = np.where(targets == 2)[0]
    plt.scatter(embeddings[inds, 0], embeddings[inds, 1], alpha=1, color=colors[2])
    inds = np.where(targets == 4)[0]
    plt.scatter(embeddings[inds, 0], embeddings[inds, 1], alpha=1, color=colors[4])
    inds = np.where(targets == 5)[0]
    plt.scatter(embeddings[inds, 0], embeddings[inds, 1], alpha=1, color=colors[5])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(mnist_classes)

def train_or_eval_model(model, net, loss_f, dataloader, train_flag=False, optimizer=None, cuda_flag=False, feature_type='text',
                        target_names=None):
    assert not train_flag or optimizer != None
    losses, preds, labels = [], [], []
    output_pz = torch.tensor([]).cuda()
    label_pz = torch.tensor([]).cuda()
    label_p_1 = torch.tensor([]).cuda()

    b_pz = torch.tensor([]).cuda()
    biaoqian_pz = torch.tensor([]).cuda()



    if train_flag:
        model.train()
        net.train()
    else:
        model.eval()
        net.eval()

    for step, data in enumerate(dataloader):
        if train_flag:
            optimizer.zero_grad()


        # roberta_fea: CLS embedding of last hidden layer in RoBERTa
        roberta_fea, xIntent, xAttr, xNeed, xWant, xEffect, xReact, oWant, oEffect, oReact, qmask, umask, label2 = [d.cuda() for d in data[:-1]] if cuda_flag else data[:-1]

        seq_lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]
        ck = torch.stack([xIntent, xAttr, xNeed, xWant, xEffect, xReact, oWant, oEffect, oReact])
        log_prob, q_c, k_ = model(roberta_fea, qmask, seq_lengths, train_flag, umask, label2, ck)
        # log_prob, a = model(roberta_fea, qmask, seq_lengths, train_flag, umask, label2, ck)
        biaoqian = torch.cat([label2[j][:seq_lengths[j]] for j in range(len(label2))])
        huayu = torch.cat([roberta_fea.transpose(0, 1)[j][:seq_lengths[j]] for j in range(len(label2))])
        huayu = nn.functional.normalize(huayu, dim=1)
        fc_b = nn.Linear(1024, 3).cuda()
        b = fc_b(huayu)

        label = torch.cat([label2[j][:seq_lengths[j]] for j in range(len(label2))])
        loss = loss_f(log_prob, label)

        preds.append(torch.argmax(log_prob, 1).cpu().numpy())
        labels.append(label.cpu().numpy())
        losses.append(loss.item())

        if train_flag:
            loss.backward()
            optimizer.step()

        # output_pz = torch.cat((a, output_pz), dim=0)
        # label_pz = torch.cat((biaoqian, label_pz), dim=0)

        b_pz = torch.cat((b, b_pz), dim=0)
        biaoqian_pz = torch.cat((biaoqian, biaoqian_pz), dim=0)

    # if epoch == 49:
    #     # plot_embeddings(output_p_f, label_p_f)
    #     # plt.show()
    #     # a = output_p[:500, :]
    #     # b = label_p_1[:500]
    #     a = output_p
    #     b = label_p_1
    #     plot_embeddings_3d(a, b)


    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
    else:
        return float('nan'), float('nan'), float('nan'), [], []

    labels = np.array(labels)
    preds = np.array(preds)

    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_accuracy = round(accuracy_score(labels, preds) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, average='weighted') * 100, 2)

    all_matrix = []
    all_matrix.append(metrics.classification_report(labels, preds, target_names=target_names, digits=4))
    all_matrix.append(["ACC"])
    for i in range(len(target_names)):
        all_matrix[-1].append("{}: {:.4f}".format(target_names[i], accuracy_score(labels[labels == i], preds[labels == i])))

    # return avg_loss, avg_accuracy, avg_fscore, all_matrix, [labels, preds], q_c, k_
    return avg_loss, avg_accuracy, avg_fscore, all_matrix, [labels, preds], b_pz, biaoqian_pz
    # return avg_loss, avg_accuracy, avg_fscore, all_matrix, [labels, preds]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--status', type=str, default='train', help='optional status: train/test')

    parser.add_argument('--feature_type', type=str, default='text', help='feature type multi/text/acouf')

    parser.add_argument('--data_dir', type=str, default='../data/iemocap/iemocap_features_roberta.pkl', help='dataset dir: iemocap_features_roberta.pkl')

    parser.add_argument('--output_dir', type=str, default='../outputs/iemocap/dialoguecrn_v1', help='saved model dir')

    parser.add_argument('--load_model_state_dir', type=str, default='../outputs/iemocap/dialoguecrn_v1/dialoguecrn_22.pkl', help='load model state dir')

    parser.add_argument('--base_model', default='LSTM', help='base model, LSTM/GRU/Linear')

    parser.add_argument('--base_layer', type=int, default=2, help='the number of base model layers,1/2')

    parser.add_argument('--epochs', type=int, default=100, metavar='E', help='number of epochs')

    parser.add_argument('--patience', type=int, default=20, help='early stop')

    parser.add_argument('--batch_size', type=int, default=32, metavar='BS', help='batch size')

    parser.add_argument('--valid_rate', type=float, default=0.1, metavar='valid_rate', help='valid rate: 0.1')

    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')

    parser.add_argument('--l2', type=float, default=0.0002, metavar='L2', help='L2 regularization weight')

    parser.add_argument('--dropout', type=float, default=0.2, metavar='dropout', help='dropout rate')

    parser.add_argument('--step_s', type=int, default=3, help='the number of reason turns at situation-level')

    parser.add_argument('--step_p', type=int, default=0, help='the number of reason turns at speaker-level')

    parser.add_argument('--gamma', type=float, default=0, help='gamma 0/1/2')

    parser.add_argument('--no_cuda', action='store_true', default=False, help='does not use GPU')

    parser.add_argument('--loss', default="FocalLoss", help='loss function: FocalLoss/NLLLoss')

    parser.add_argument('--class_weight', action='store_true', default=True, help='use class weights')

    parser.add_argument('--tensorboard', action='store_true', default=False, help='enables tensorboard log')

    parser.add_argument('--seed', type=int, default=0, help='random seed')

    args = parser.parse_args()
    print(args)

    epochs, batch_size, status, output_path, data_path, base_model, base_layer, feature_type = \
        args.epochs, args.batch_size, args.status, args.output_dir, args.data_dir, args.base_model, args.base_layer, args.feature_type
    cuda_flag = torch.cuda.is_available() and not args.no_cuda
    reason_steps = [args.step_s, args.step_p]

    if args.tensorboard:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter()

    # IEMOCAP dataset
    n_classes, n_speakers, hidden_size, input_size = 6, 2, 128, None
    target_names = ['hap', 'sad', 'neu', 'ang', 'exc', 'fru']
    # perform 1/weight according to the weight of each label in training data
    class_weights = torch.FloatTensor([1 / 0.087178797, 1 / 0.145836136, 1 / 0.229786089, 1 / 0.148392305, 1 / 0.140051123, 1 / 0.24875555])

    if feature_type in ['text']:
        input_size = 1024
    else:
        print('Error: feature_type not set.')
        exit(0)

    seed_everything(seed=args.seed)
    model = DialogueCRN(base_model=base_model,
                        base_layer=base_layer,
                        input_size=input_size,
                        hidden_size=hidden_size,
                        n_speakers=n_speakers,
                        n_classes=n_classes,
                        dropout=args.dropout,
                        cuda_flag=cuda_flag,
                        reason_steps=reason_steps)

    net = bianhuan()
    net.cuda()
    if cuda_flag:
        print('Running on GPU')
        # torch.cuda.set_device(3)  # test
        class_weights = class_weights.cuda()
        model.cuda()

    else:
        print('Running on CPU')

    name = 'DialogueCRN'
    print('{} with {} as base model.'.format(name, base_model))
    print("The model have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    print('Running on the {} features........'.format(feature_type))
    for name, param in model.named_parameters():
        print(name, param.size(), param.requires_grad)

    if args.loss == 'FocalLoss':
        # FocalLoss
        loss_f = FocalLoss(gamma=args.gamma, alpha=class_weights if args.class_weight else None)
    else:
        # NLLLoss
        loss_f = nn.NLLLoss(class_weights if args.class_weight else None)

    train_loader, valid_loader, test_loader = get_IEMOCAP_bert_loaders(path=data_path, batch_size=batch_size, num_workers=0, valid_rate=args.valid_rate)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)


    if status == 'train':
        all_test_fscore, all_test_acc = [], []
        best_epoch, best_epoch2, patience, best_eval_fscore, best_eval_loss = -1, -1, 0, 0, None
        patience2 = 0
        for e in range(epochs):
            start_time = time.time()

            train_loss, train_acc, train_fscore, _, _, b1, biaoqian1 = train_or_eval_model(model=model, net=net, loss_f=loss_f, dataloader=train_loader, train_flag=True,
                                                                            optimizer=optimizer, cuda_flag=cuda_flag, feature_type=feature_type,
                                                                            target_names=target_names)
            valid_loss, valid_acc, valid_fscore, _, _, b2, biaoqian2 = train_or_eval_model(model=model, net=net, loss_f=loss_f, dataloader=valid_loader, cuda_flag=cuda_flag,
                                                                            feature_type=feature_type, target_names=target_names)
            test_loss, test_acc, test_fscore, test_metrics, _, b3, biaoqian3 = train_or_eval_model(model=model, net=net, loss_f=loss_f, dataloader=test_loader,
                                                                                    cuda_flag=cuda_flag, feature_type=feature_type, target_names=target_names)
            all_test_fscore.append(test_fscore)
            all_test_acc.append(test_acc)

            # if args.valid_rate > 0:
            #     eval_loss, _, eval_fscore = valid_loss, valid_acc, valid_fscore
            # else:
            eval_loss, _, eval_fscore = test_loss, test_acc, test_fscore
            if e == 0 or best_eval_fscore < eval_fscore:
                patience = 0
                best_epoch, best_eval_fscore = e, eval_fscore
                if not os.path.exists(output_path): os.makedirs(output_path)
                save_model_dir = os.path.join(output_path, 'f1_{}_{}.pkl'.format(name, e).lower())
                torch.save(model.state_dict(), save_model_dir)
            else:
                patience += 1

            if best_eval_loss is None:
                best_eval_loss = eval_loss
                best_epoch2 = 0
            else:
                if eval_loss < best_eval_loss:
                    best_epoch2, best_eval_loss = e, eval_loss
                    patience2 = 0
                    if not os.path.exists(output_path): os.makedirs(output_path)
                    save_model_dir = os.path.join(output_path, 'loss_{}_{}.pkl'.format(name, e).lower())
                    torch.save(model.state_dict(), save_model_dir)

                else:
                    patience2 += 1

            if args.tensorboard:
                writer.add_scalar('train: accuracy/f1/loss', train_acc / train_fscore / train_loss, e)
                writer.add_scalar('valid: accuracy/f1/loss', valid_acc / valid_fscore / valid_loss, e)
                writer.add_scalar('test: accuracy/f1/loss', test_acc / test_fscore / test_loss, e)
                writer.close()

            print(
                'epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'. \
                    format(e, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss,
                           test_acc, test_fscore,
                           round(time.time() - start_time, 2)))
            print(test_metrics[0])
            print(test_metrics[1])

            # if patience >= args.patience and patience2 >= args.patience:
            #     print('Early stoping...', patience, patience2)
            #     break

            if e == 99:
                plot_embeddings_3d(b1, biaoqian1)
                # 保存图形为PDF文件
                plt.savefig('sandiantu1.pdf', format='pdf')

                plot_embeddings_3d(b2, biaoqian2)
                # 保存图形为PDF文件
                plt.savefig('sandiantu2.pdf', format='pdf')

                plot_embeddings_3d(b3, biaoqian3)
                # 保存图形为PDF文件
                plt.savefig('sandiantu3.pdf', format='pdf')
                plt.show()


        print('Final Test performance...')
        print('Early stoping...', patience, patience2)
        print('Eval-metric: F1, Epoch: {}, best_eval_fscore: {}, Accuracy: {}, F1-Score: {}'.format(best_epoch,
                                                                                                    best_eval_fscore,
                                                                                                    all_test_acc[
                                                                                                        best_epoch] if best_epoch >= 0 else 0,
                                                                                                    all_test_fscore[
                                                                                                        best_epoch] if best_epoch >= 0 else 0))
        print('Eval-metric: Loss, Epoch: {}, Accuracy: {}, F1-Score: {}'.format(best_epoch2,
                                                                                all_test_acc[
                                                                                    best_epoch2] if best_epoch2 >= 0 else 0,
                                                                                all_test_fscore[

                                                                                    best_epoch2] if best_epoch2 >= 0 else 0))
        # viz = visdom.Visdom()
        # if e == 1:
        #     # plot_embeddings(output_p_f, label_p_f)
        #     # plt.show()
        #     # a = output_p[:500, :]
        #     # b = label_p_1[:500]
        #     a = output_p
        #     b = label_p_1
        #     plot_embeddings_3d(a, b)
        # if e == 1:
        #     # plot_embeddings(output_p_f, label_p_f)
        #     # plt.show()
        #     # a = output_p[:500, :]
        #     # b = label_p_1[:500]
        #     a = output_p_2
        #     b = label_p_1_2
        #     plot_embeddings_3d(a, b)
        # if e == 1:
        #     # plot_embeddings(output_p_f, label_p_f)
        #     # plt.show()
        #     # a = output_p[:500, :]
        #     # b = label_p_1[:500]
        #     a = output_p_3
        #     b = label_p_1_3
        #     plot_embeddings_3d(a, b)

    elif status == 'test':
        start_time = time.time()
        model.load_state_dict(torch.load(args.load_model_state_dir))
        test_loss, test_acc, test_fscore, test_metrics, test_outputs = train_or_eval_model(model=model, loss_f=loss_f,
                                                                                           dataloader=test_loader,
                                                                                           cuda_flag=cuda_flag,
                                                                                           feature_type=feature_type)

        if args.tensorboard:
            writer.add_scalar('test: accuracy/loss', test_acc / test_loss, 0)
        print('test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'.format(test_loss, test_acc, test_fscore,
                                                                                  round(time.time() - start_time, 2)))
        print(test_metrics[0])
        print(test_metrics[1])
    else:
        print('the status must be one of train/test')
        exit(0)



