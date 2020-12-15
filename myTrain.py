import  random
import logging
from torch.utils.data import SequentialSampler, RandomSampler, DataLoader
from  tqdm import  tqdm

from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from utils.config import *
from utils.data_utils import *
from utils.evaluate_utils import *
from utils.other_utils import del_file, cleanup
from models.model import  transformer_model

def evaluate(model, dev_dataloader):
    model.eval()
    logger.info("Running Evaluation...")

    all_pred_labels, all_gold_labels = [], []
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

    score_m = match_m(all_pred_labels, all_gold_labels)
    score_avg = np.mean(list(score_m.values()))
    logger.info("Evaluation Accuracy: ")
    logger.info("1:{:0.4f} 5:{:0.4f} 10:{:0.4f}".format(score_m[1], score_m[5], score_m[10]))
    logger.info("Average: {:0.4f}".format(score_avg))

    return score_avg, score_m

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
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    #Set Random Seed
    if args.fix_seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    #Loading data
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_data,train_word_id,_ = read_data(args.train_file,tokenizer)
    train_dataset=Data(train_data,tokenizer)
    train_sampler= RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, collate_fn=train_dataset.collate_fn,
                                  batch_size= args.batch_size, num_workers=8)
    dev_data,dev_word_id,_ = read_data(args.dev_file, tokenizer)
    dev_dataset = Data(dev_data, tokenizer)
    dev_dataloader = DataLoader(dev_dataset, sampler=SequentialSampler(dev_dataset), collate_fn=dev_dataset.collate_fn
                                , batch_size=args.batch_size, num_workers=8)
    logger.info('Loaded data')

    #Initialize the model
    model=transformer_model(args.model_name, device, args.dropout_prob).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, eps=args.epsilon)

    # Create the learning rate scheduler.
    total_steps = len(train_dataloader) * args.epoch
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=args.warm_up_steps,  # Default value in run_glue.py
                                                num_training_steps=total_steps)
    ####################################################################################
    # Training and evaluating
    ####################################################################################
    early_stop_count=0
    opt_global_step=0
    max_accuracy, max_match=0, None
    pre_saved_model = ""
    for epoch_i in range(0, args.epoch):
        logger.info('======== Epoch {:} / {:} ========'.format(epoch_i + 1, args.epoch))
        logger.info('Training...')

        #shuffle the data

        total_loss = 0
        model.train()
        pbar=tqdm(enumerate(train_dataloader), total=len(train_dataloader), ncols=80)
        for step, batch in pbar:
            sents_tokens, sents_len, tokens_start, labels,sents_other_feats= \
                batch['sents_tokens'].cuda(),batch['sents_len'].cuda(),batch['sents_tokens_start'].cuda(),\
                batch['sents_label'].cuda(), batch['sents_other_feats'].cuda()
            output = model(sents_tokens, tokens_start, sents_other_feats, sents_len, labels)
            loss, _ = output
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            opt_global_step+=1

        avg_train_loss = total_loss / len(train_dataloader)

        #Evaluate
        accuracy, score_m = evaluate(model, dev_dataloader)
        early_stop_count+=1
        if (accuracy > max_accuracy):
            early_stop_count=0
            max_accuracy = accuracy
            max_score_m = score_m
            if(del_file(pre_saved_model)):
                logging.info("Previous best model deleted")
            saved_model_name = "P:" + (args.model_name).split("/")[-1]+"-b"+str(args.batch_size)+"-lr"+\
                               str(args.lr)+"-dr"+str(args.dropout_prob)+"-wup"+str(args.warm_up_steps)+\
                               "-{:.4f}".format(max_accuracy)
            torch.save(model.state_dict(), os.path.join(args.model_save_dir, saved_model_name))
            pre_saved_model=os.path.join(args.model_save_dir,saved_model_name)
            logger.info("Model saved. Best accuracy = {:.4f}".format(max_accuracy))

        if early_stop_count >= args.early_stop:
            break
    logger.info("Training complete!")
    logger.info('Max accuracy\n'+ str(max_accuracy))




