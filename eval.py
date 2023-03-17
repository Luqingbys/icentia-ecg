import argparse
import numpy as np
import matplotlib.pylab as plt
import sys
import pandas as pd
import sklearn
import sklearn.model_selection
import sklearn.neighbors
import sklearn.linear_model
import sklearn.ensemble
import gzip
import utils
import encoders
import collections
from tqdm import tqdm
from mlp import MLP_train, MLP
import pytorch_models

parser = argparse.ArgumentParser()
parser.add_argument('-results_file', nargs='?', default="G:\深度学习\医疗\icentia-ecg\icentia-ecg\\results.csv",
                    help='file to load and write results to')
# parser.add_argument('-results_file', nargs='?', default="results.csv",
#                     help='file to load and write results to')
parser.add_argument('-label_type', nargs='?',
                    default="btype", help='what to predict')
parser.add_argument('-num_examples', nargs='?',
                    type=int, default=1200, help='')
parser.add_argument('-num_trials', nargs='?', type=int, default=6, help='')
parser.add_argument('-sequences_file', nargs='?',
                    default="test_emb.csv.gz",  help='File with sequences')
parser.add_argument('-labels_file', nargs='?',
                    default="test_labels.csv.gz", help='')
parser.add_argument('-embeddings_file', type=str, default='test_emb.csv.gz', help='')
parser.add_argument('-model', type=str, default="knn", choices=[
                    "knn", "mlp", "lr", "adaboost", "conv-resnet", "conv-basic"], help='Model to evaluate embeddings with.')
parser.add_argument('-encode_method', type=str, default=None, choices=[o for o in dir(
    encoders) if not o.startswith("_")], help='to encode the signals on the fly')
    # dir(encoders): [convautoencoder, none, rand, convautoencoder_random, pca, ...]
args = parser.parse_args()

print(args)


def evaluate(model_name, num_examples, label_type, seed, encode_method=None):
    ''' 
    model_name: 模型名字
    num_examples: 默认值为1200
    label_type: 默认值为btype
    '''

    print("Generating subset", seed)

    # 读取数据集, labels_file: test_labels.csv.gz
    data, labels = utils.getSubset(num_examples,
                                   embeddings_file=args.embeddings_file,
                                   labels_file=args.labels_file,
                                   seed=seed,
                                   balanced=label_type) 
    # data: pandas.DataFrame, (180, 2049), 180条样本
    # label: pandas.DataFrame, [sample  segment   frame  btype  rtype], (180, 5)
    # print('labels: ', labels)

    if encode_method != None:
        # 如果使用了编码器（特征学习），就导入模块encoders的属性
        enc = getattr(encoders, encode_method)() # enc是模块encoders下的属性，比如convautoencoder
        # print("Encoder:", enc)
        newdata = []
        for emb in data.values: # emb: ndarray, (2049, )
            # 原始数据通过enc.encode，也即convautoencoder的encode方法，这个方法实际上调用的是model模块下的Autoencoder
            # print('emb: ', type(emb), emb.shape) 
            newdata.append(enc.encode(emb))
        data = np.asarray(newdata) # data: (180, 100)
        # print('data: ', data.shape)

    print(collections.Counter(labels[label_type])) # Counter({2: 60, 1: 60, 0: 60})

    # X, X_test; y, y_test分别是训练集和测试集的特征和标签，并且，该特征是通过了自编码器之后的编码结果（具有比较好的特征）
    X, X_test, y, y_test = \
        sklearn.model_selection.train_test_split(data, labels[label_type],
                                                 train_size=len(labels)//2,
                                                 test_size=len(labels)//2,
                                                 stratify=labels[label_type],
                                                 random_state=seed) # X: (90, 100)  X_test: (90, 100)

    # 选择不同的模型
    if model_name == "knn":
        model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=3)
    elif model_name == "lr":
        model = sklearn.linear_model.LogisticRegression(multi_class="auto")
    elif model_name == "adaboost":
        model = sklearn.ensemble.AdaBoostClassifier()
    elif model_name == "mlp":
        network = pytorch_models.MLP(
            in_channels=len(X.values[0]),
            out_channels=1000,
            n_classes=len(set(y)),
            seed=seed)
        model = pytorch_models.PyTorchModel(
            network, device="cuda", batch_size=32, n_epoch=40, seed=seed)
    elif model_name == "conv-basic":
        network = pytorch_models.CNN(
            in_channels=1,
            out_channels=10,
            n_layers=5,
            stride=2,
            kernel=50,
            final_layer=120,
            n_classes=len(set(y)),
            seed=seed)
        model = pytorch_models.PyTorchModel(
            network, device="cuda", batch_size=32, n_epoch=40, seed=seed)
    elif model_name == "conv-resnet":
        network = pytorch_models.ResNet1D(
            in_channels=1,
            base_filters=128,  # 64 for ResNet1D, 352 for ResNeXt1D
            kernel_size=16,
            stride=2,
            groups=32,
            n_block=48,
            n_classes=len(set(y)),
            downsample_gap=6,
            increasefilter_gap=12,
            use_do=True,
            seed=seed)
        model = pytorch_models.PyTorchModel(
            network, device="cuda", batch_size=32, n_epoch=40, seed=seed)
    else:
        print("Unknown model")
        sys.exit()
    print(model)
    model.fit(X, y.values.flatten())
    y_pred = model.predict(X_test)
    bacc = sklearn.metrics.balanced_accuracy_score(
        y_test.values.flatten(), y_pred)
    print("   Run {} ".format(seed) + model_name +
          ", label_type: {}".format(label_type) + ", Balanced Accuracy Test: {}".format(bacc))

    return bacc


results = pd.read_csv(args.results_file)


for seed in range(0, args.num_trials):
    res = {"model": args.model,
           "num_examples": int(args.num_examples),
           "label_type": args.label_type,
           "seed": int(seed),
           "encode_method": args.encode_method}
    # if (len(results) > 0) and (len(pd.concat([pd.DataFrame(res, index=[0]), results])) > 0):
    # if (len(results) > 0) and (len(pd.merge(pd.DataFrame(res, index=[0]), results)) > 0):
        # print("already done: ", res)
        # continue
    print("running: ", res)
    bacc = evaluate(args.model,
                    num_examples=args.num_examples,
                    label_type=args.label_type,
                    seed=seed,
                    encode_method=args.encode_method)
    res["bacc"] = bacc
    results = results.append(res, ignore_index=True)
    results.to_csv(args.results_file, index=False)


results2 = results.copy().fillna("")
results2 = results2[results2.label_type == "btype"]
results2 = results2[results2.num_examples > 50]
results2 = results2[results2.encode_method != "fft"]
results2.num_examples = results2.num_examples.astype(int)
print(results2.groupby(["label_type", "num_examples", "model", "encode_method"]).agg({
    'bacc': ['mean', 'std', 'count']}).round(2).to_csv())
