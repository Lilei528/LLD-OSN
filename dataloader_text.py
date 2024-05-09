import random
from transformers import BertTokenizer



def generate_train_dataset(filepath):
    """ Read corpus from given file path and split it into train and dev parts
    Args:
        filepath: file path
        sent_vocab: sentence vocab
        tag_vocab: tag vocab
        train_proportion: proportion of training data
    Returns:
        train_data: data for training, list of tuples, each containing a sentence and corresponding tag.
        dev_data: data for development, list of tuples, each containing a sentence and corresponding tag.
    """
    all_lines=get_data(filepath)
    label=[]
    text=[]
    for i in all_lines:
        tag,sent=i.split('\t')[0],i.split('\t')[1].replace('\n','')
        label.append(tag)
        text.append(sent)

    return text,label


class process_data(object):

    def __init__(self,data,batch_size=32, shuffle=True):
        self.org_text = data[0]
        self.org_label = data[1]
        self.data_size = len(self.org_text)
        self.indices = list(range(self.data_size))
        if shuffle:
            random.shuffle(self.indices)
        self.batch_num = (self.data_size + batch_size - 1) // batch_size
        self.label = [int(self.org_label[idx]) for idx in self.indices]



    def batch_iter(self,padding_len,tokenizer, batch_size=32, shuffle=True):
        for i in range(self.batch_num):
            batch_text = [self.org_text[idx] for idx in self.indices[i * batch_size: (i + 1) * batch_size]]
            batch_label= [int(self.org_label[idx]) for idx in self.indices[i * batch_size: (i + 1) * batch_size]]
            batch_text_2 =[int(self.org_label[idx]) for idx in self.indices[i * batch_size: (i + 1) * batch_size]]
            mask = []
            for index,t in enumerate(batch_text):
                this_mask=[False]*padding_len
                fenci=tokenizer.tokenize(repr(t))
                if fenci==None:
                    print(t)
                #padding
                if len(fenci)>=(padding_len-2):
                    fenci = fenci[:padding_len - 2]
                    fenci.insert(0, '[CLS]')
                    fenci.append('[SEP]')

                elif len(fenci)<(padding_len-2):
                    fenci.insert(0, '[CLS]')
                    fenci.append('[SEP]')
                    fenci.extend(['[PAD]']*( (padding_len) -len(fenci)  ))
                    for t1,ttt in enumerate(fenci):
                        if ttt =='[PAD]':
                            this_mask[t1]=True
                #add special token


                encod1=tokenizer.convert_tokens_to_ids(fenci)


                batch_text[index]=encod1
                encod2 = tokenizer.convert_tokens_to_ids(fenci)

                batch_text_2[index]=encod2
                mask.append(this_mask)

            yield {'index':[idx for idx in self.indices[i * batch_size: (i + 1) * batch_size]],'data':(batch_text,batch_text_2),'label':batch_label,'mask':mask}



def get_data(file_path):
    f=open(file_path,'r',encoding='utf-8')
    all_line=f.readlines()
    f.close()
    return all_line




