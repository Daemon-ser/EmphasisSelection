import torch
from utils.config import *
from utils.data_utils import *
from utils.evaluate_utils import *
from torch.utils.data import  SequentialSampler,DataLoader
import logging
import random
from models.model import transformer_model
from transformers import AutoTokenizer
from utils.io_utils import write_results,convert_to_slide_format
from tqdm import tqdm

def dev(model, dev_dataloader):
    all_pred_labels, all_gold_labels, all_label_lens= [], [],[]
    pbar = tqdm(enumerate(dev_dataloader), total=len(dev_dataloader), ncols=80)
    for i,batch in pbar:
        sents_tokens, sents_len, tokens_start, labels, sents_other_feats = \
            batch['sents_tokens'].cuda(), batch['sents_len'].cuda(), batch['sents_tokens_start'].cuda(), batch['sents_label'].cuda(), batch[
                'sents_other_feats'].cuda()

        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():
            output = model(sents_tokens, tokens_start, sents_other_feats, sents_len, labels)
        _, pred_labels = output
        pred_labels = pred_labels.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        pred_labels, labels = fix_padding(pred_labels, labels, sents_len)
        all_pred_labels.extend(pred_labels)
        all_gold_labels.extend(labels)
        all_label_lens.extend(sents_len.detach().cpu().numpy())
    return all_pred_labels, all_gold_labels,all_label_lens



if __name__=="__main__":

    #Logger Setting
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    fileHandler = logging.FileHandler(os.path.join(args.log_file))
    logger.addHandler(fileHandler)
    logger.info(args)

    # CUDA setup
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    #Set Random Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    #Loading data

    all_pred_labels, gold_labels, label_lens=[],[],None
    if args.bert_test_model_list is not None:
        tokenizer = AutoTokenizer.from_pretrained('xlnet-large-cased')
        dev_data, dev_words_id, dev_words = read_data(args.dev_file, tokenizer)
        dev_dataset = Data(dev_data, tokenizer)
        dev_dataloader = DataLoader(dev_dataset, sampler=SequentialSampler(dev_dataset),
                                    collate_fn=dev_dataset.collate_fn, batch_size=args.batch_size, num_workers=8)
        logger.info('Data Loaded')
        for i, model_name in enumerate(args.bert_test_model_list):
            model=transformer_model('xlnet-large-cased', device, args.dropout_prob).to(device)
            model.load_state_dict(torch.load(os.path.join(args.model_save_dir, model_name)))
            model.eval()
            pred_labels, gold_labels, label_lens= dev(model,dev_dataloader)
            all_pred_labels.append(pred_labels)

    if args.xlm_test_model_list is not None:
        tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')
        dev_data, dev_words_id, dev_words = read_data(args.dev_file, tokenizer)
        dev_dataset = Data(dev_data, tokenizer)
        dev_dataloader = DataLoader(dev_dataset, sampler=SequentialSampler(dev_dataset),
                                    collate_fn=dev_dataset.collate_fn, batch_size=args.batch_size, num_workers=8)
        logger.info('Data Loaded')
        for i, model_name in enumerate(args.xlm_test_model_list):
            model=transformer_model('xlm-roberta-large', device, args.dropout_prob).to(device)
            model.load_state_dict(torch.load(os.path.join(args.model_save_dir, model_name)))
            model.eval()
            pred_labels, gold_labels, label_lens= dev(model,dev_dataloader)
            all_pred_labels.append(pred_labels)



    mean_pred_label=np.mean(all_pred_labels,axis=0)
    mean_pred_label, _ = fix_padding(mean_pred_label,mean_pred_label,label_lens)
    score_m = match_m(mean_pred_label, gold_labels)
    score_avg = np.mean(list(score_m.values()))
    logger.info("Evaluation Accuracy: ")
    logger.info("1:{:0.4f} 5:{:0.4f} 10:{:0.4f}".format(score_m[1], score_m[5], score_m[10]))
    logger.info("Average: {:0.4f}".format(score_avg))


