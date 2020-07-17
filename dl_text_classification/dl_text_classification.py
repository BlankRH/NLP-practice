import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import gc

import argparse

from gensim.models import Word2Vec, KeyedVectors

from read_data import get_data


class textCNN(nn.Module):
    def __init__(self, args):
        super(textCNN, self).__init__()
        self.args = args
        channel = 1
        c_num = args.class_num
        v = args.embed_num
        dim = args.embed_dim
        if args.load_vectors:
            weights = torch.FloatTensor(args.vectors)
            self.embedding = nn.Embedding.from_pretrained(weights, max_norm=args.max_norm, freeze=False)
        else:
            self.embedding = nn.Embedding(v, dim, max_norm=args.max_norm)
        k_sizes = args.kernel_sizes
        k_dim = args.kernel_num
        self.convs = nn.ModuleList([nn.Conv2d(channel, k_dim, (k, dim)) for k in k_sizes])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(len(k_sizes) * k_dim, c_num)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(line, line.size(2)).squeeze(2) for line in x]

        x = torch.cat(x, 1)
        x = self.dropout(x)
        logit = self.fc(x)
        return logit


def train(train_iter, dev_iter, model, args):

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.constraint)

    steps = 0
    best_acc = 0
    last_step = 0
    model.train()
    for epoch in range(args.epochs):
        for batch in train_iter:
            feature, target = batch.Phrase, batch.Sentiment
            feature.t_()
            feature, target = feature.cuda(), target.cuda()
            optimizer.zero_grad()
            logit = model(feature)
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()
            steps += 1

            if steps % args.log_interval == 0:
                res = torch.max(logit, 1)[1].view(target.size())
                corrects = (res.data == target.data).sum()
                acc = corrects*100.0/batch.batch_size
                print('\rBatch[{}] - loss: {:.6f} acc: {:.4f}$({}/{})'.format(steps,
                                                                            loss.data.item(),
                                                                            acc,
                                                                            corrects,
                                                                            batch.batch_size))

            if steps % args.test_interval == 0:
                dev_acc = eval(dev_iter, model, args)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best:
                        save(model, args.save_dir, 'best', steps)
                else:
                    if steps - last_step >= args.early_stop:
                        print('early stop by {} steps.'.format(args.early_stop))
                print()
            if steps % args.save_interval == 0:
                save(model, args.save_dir, 'snapshot', steps)

def eval(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.Phrase, batch.Sentiment
        feature.t_()
        feature, target = feature.cuda(), target.cuda()
        logit = model(feature)
        loss = F.cross_entropy(logit, target, reduction='sum')
        avg_loss += loss.item()
        corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss /= size
    acc = 100.0 * corrects/size
    print('Evaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) '.format(avg_loss,
                                                                       acc,
                                                                       corrects,
                                                                       size))
    return acc

def save(model, save_dir, save_prefix, steps):
    if not os.path.join(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN text classifier')
    parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
    parser.add_argument('-epochs', type=int, default=256, help='number of epochs for train [default: 256]')
    parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
    parser.add_argument('-log-interval', type=int, default=100,
                        help='how many steps to wait before logging training status [default: 1]')
    parser.add_argument('-test-interval', type=int, default=100,
                        help='how many steps to wait before testing [default: 100]')
    parser.add_argument('-save-interval', type=int, default=2000,
                        help='how many steps to wait before saving [default:500]')
    parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
    parser.add_argument('-early-stop', type=int, default=1000,
                        help='iteration numbers to stop without performance increasing')
    parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
    # data
    parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
    # model
    parser.add_argument('-dropout', type=float, default=0.4, help='the probability for dropout [default: 0.5]')
    parser.add_argument('-constraint', type=float, default=0.0, help='l2 constraint of parameters [default: 3.0]')
    parser.add_argument('-max-norm', type=float, default=20, help='max sentence size [default: 20]')
    parser.add_argument('-embed-dim', type=int, default=300, help='number of embedding dimension [default: 128]')
    parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
    parser.add_argument('-kernel-sizes', type=str, default=(2, 3, 4, 5),
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
    # device
    parser.add_argument('-device', type=int, default=0,
                        help='device to use for iterate data, -1 mean cpu [default: -1]')
    parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
    parser.add_argument('-load-vectors', type=bool, default=True, help='pre-trained word embedding vectors [default: True]')

    args = parser.parse_args()

    vocab, train_iter, dev_iter, test_iter = get_data(args.batch_size)
    vocab_size = len(vocab)


    args.class_num = 5

    vocab_list = vocab.itos
    args.embed_num = len(vocab_list)

    args.load_vectors = True

    if args.load_vectors:
        w2v_model = KeyedVectors.load_word2vec_format('data\\GoogleNews-vectors-negative300.bin',
                                                  binary=True)
        for i in range(len(vocab_list)):
            if vocab_list[i] not in w2v_model:
                vocab_list[i] = 'UNK'
        args.vectors = [w2v_model[x] for x in vocab_list]
        del w2v_model
        gc.collect()

    cnn = textCNN(args)
    args.snapshot = 'snapshot\\best_steps_3900_w2v.pt'
    if args.snapshot is not None:
        print('\nLoading model from {}...'.format(args.snapshot))
        cnn.load_state_dict(torch.load(args.snapshot))

    torch.cuda.set_device(args.device)
    cnn = cnn.cuda()

    try:
        train(train_iter, dev_iter, cnn, args)
    except KeyboardInterrupt:
        print('\n' + '-' * 89)
        print('Exiting from training early')
