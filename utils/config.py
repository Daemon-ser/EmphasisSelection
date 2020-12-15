import argparse
import warnings
warnings.filterwarnings("ignore")


parser=argparse.ArgumentParser()

parser.add_argument("--all_file", default='datasets/data_pos/all.txt', required=False)
parser.add_argument("--train_file", default='datasets/data_pos/train.txt', required=False)
parser.add_argument("--dev_file", default='datasets/data_pos/dev.txt', required=False)
parser.add_argument("--test_file", default='datasets/data_pos/test_data.txt', required=False)

parser.add_argument("--model_name", default='xlm-roberta-large', required=False)
parser.add_argument("--epoch", type=int, default=100, required=False)
parser.add_argument("--batch_size", type=int, default=1, required=False)
parser.add_argument("--dropout_prob", type=float, default=0.3, required=False)
parser.add_argument("--early_stop",type=int, default=5, required=False)
parser.add_argument("--to_freeze", action='store_true', required=False)
parser.add_argument("--add_features", type=int, default=0, required=False)

parser.add_argument("--no_cuda", action='store_true', required=False)
parser.add_argument("--seed", type=int, default=1234, required=False)
parser.add_argument("--fix_seed", action='store_true', required=False)

parser.add_argument("--lr", type=float, default=2e-5, required=False)
parser.add_argument("--warm_up_steps", type=int, default=0, required=False)
parser.add_argument("--epsilon", type=float, default=1e-8, required=False)

parser.add_argument("--log_file", default="all_outputs/logs/xlnet.txt", required=False)
parser.add_argument("--model_save_dir", default="all_outputs/saved_models/xlm-roberta-large/", required=False)
parser.add_argument("--test_result_dir", default="all_outputs/result", required=False)
parser.add_argument("--test_model_name",  required=False)
parser.add_argument("--test_model_list", nargs='*')
parser.add_argument("--xlm_test_model_list", nargs='*')
parser.add_argument("--bert_test_model_list", nargs='*')

args=parser.parse_args()

