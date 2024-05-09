import json

from sklearn.datasets import fetch_20newsgroups
def make_20newsgroup(train_dir='./dataset/20newsgroup/train.csv',test_dir='./dataset/20newsgroup/test.csv',labeltoid='./dataset/20newsgroup/labeltoid.json'):
    # 生成训练和测试csv文件
    news_20_train=fetch_20newsgroups(data_home='./dataset/20newsgroup', # 文件下载的路径
                   subset='train', # 加载那一部分数据集 train/test
                   shuffle=True,  # 将数据集随机排序
                    remove= ('headers', 'footers', 'quotes'),
                   download_if_missing=True # 如果没有下载过，重新下载
                   )
    train_data = [{'target': target, 'data': text.replace('\n',' ').replace('\t',' ').replace('\r',' ')} for target, text in
                  zip(news_20_train.target, news_20_train.data)]
    save_to_file(train_data, train_dir)
    news_20_test=fetch_20newsgroups(data_home='./dataset/20newsgroup', # 文件下载的路径
                   subset='test', # 加载那一部分数据集 train/test
                   shuffle=True,  # 将数据集随机排序
                    remove= ('headers', 'footers', 'quotes'),
                   download_if_missing=True # 如果没有下载过，重新下载
                   )

    test_data = [{'target': target, 'data': text.replace('\n',' ').replace('\t',' ').replace('\r',' ')} for target, text in
                  zip(news_20_test.target, news_20_test.data)]
    save_to_file(test_data, test_dir)
    # 生成label和对应的id
    save_labeltoid(news_20_train.target_names,labeltoid)



def save_labeltoid(data, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        idtolabel={  i:j for i ,j in enumerate(data)}
        labeltoid = { j:i for i, j in enumerate(data)}
        labeldata={'idtolabel':idtolabel,"labeltoid":labeltoid}
        json.dump(labeldata,file)


def save_to_file(data, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for item in data:
            file.write(f"{item['target']}\t{item['data']}\n")
