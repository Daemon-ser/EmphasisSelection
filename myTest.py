import torch
from utils.config import *
from utils.data_utils import *
from utils.evaluate_utils import *
from utils.io_utils import write_results,convert_to_slide_format
from torch.utils.data import  SequentialSampler,DataLoader
import logging
import random
from models.model import transformer_model
from transformers import AutoTokenizer

from tqdm import tqdm

def test(model, dev_dataloader):
    logger.info("Running Evaluation...")
    all_pred_labels , all_lens= [],[]
    pbar = tqdm(enumerate(dev_dataloader), total=len(dev_dataloader), ncols=80)
    for i,batch in pbar:
        sents_tokens, sents_len, tokens_start, sents_other_feats = \
            batch['sents_tokens'].cuda(), batch['sents_len'].cuda(), batch['sents_tokens_start'].cuda(), batch[
                'sents_other_feats'].cuda()
        with torch.no_grad():
            output = model(sents_tokens, tokens_start, sents_other_feats, sents_len)
        _, pred_labels = output
        pred_labels = pred_labels.detach().cpu().numpy()
        #pred_labels, pred_labels = fix_padding(pred_labels, pred_labels, sents_len)
        all_pred_labels.extend(pred_labels)
        all_lens.extend(sents_len.detach().cpu().numpy())

    return all_pred_labels,all_lens



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
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    test_data, test_words_id, test_words = read_test_data(args.test_file, tokenizer)
    test_dataset = TestData(test_data,tokenizer)
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset),
                                 collate_fn=test_dataset.collate_fn, batch_size=args.batch_size, num_workers=8)
    logger.info('Data Loaded')
    all_pred_labels, label_lens=[],None
    for model_name in args.test_model_list:
        model=transformer_model(args.model_name, device, args.dropout_prob).to(device)
        model.load_state_dict(torch.load(os.path.join(args.model_save_dir,model_name)))
        model.eval()
        pred_labels, label_lens= test(model,test_dataloader)
        all_pred_labels.append(pred_labels)
    mean_pred_label=np.mean(all_pred_labels,axis=0)
    mean_pred_label, _ =fix_padding(mean_pred_label,mean_pred_label,label_lens)
    slides_words_id, slides_words, slides_mean_pred_label=convert_to_slide_format(test_words_id, test_words, mean_pred_label)
    write_results(slides_words_id,slides_words, slides_mean_pred_label, os.path.join(args.test_result_dir, 'bert_kfold_feat_1.txt'))
