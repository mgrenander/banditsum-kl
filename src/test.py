from pyrouge import Rouge155
import torch
import os, sys, logging, tempfile, argparse, pickle
from shutil import copyfile, rmtree
from tqdm import tqdm
from corenlp import CoreNLPClient
from collections import defaultdict
from model import SimpleRNN
from helper import process_text, convert_tokens_to_ids, config, Vocab
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')


def make_summaries(args):
    # Tokenizer
    corenlp_tokenizer = CoreNLPClient(annotators=['ssplit', 'tokenize'], stdout=sys.stderr, timeout=10000, max_char_length=1020000)
    corenlp_tokenizer.start()

    # Load model
    logging.info("Loading model {}".format(args.model_file))
    rewards = {"train": None, "train_single": None, "dev": None, "dev_single": None}
    model = SimpleRNN(args, rewards)
    model.cuda()

    checkpoint = torch.load(args.model_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # For counting positions
    pos_counts = defaultdict(int)

    # Folder details
    article_dir = os.path.join(args.data_dir, 'articles')

    logging.info("Starting evaluation.")
    try:
        with torch.no_grad():
            for i in tqdm(range(len(os.listdir(article_dir)))):
                article_name = str(i).rjust(6, '0') + "_article.txt"
                with open(os.path.join(article_dir, article_name), 'r') as art_file:
                    doc_sents = process_text(corenlp_tokenizer, art_file)
                    doc_ids = convert_tokens_to_ids(doc_sents, args)

                    # Write model hypothesis to file
                    summary_idx = model(doc_ids.cuda())

                    hyp_file = str(i).rjust(5, '0') + '_hypothesis.txt'
                    with open(os.path.join(args.hyp_dir, hyp_file), 'w') as f:
                        hyp_sents = [doc_sents[j] for j in summary_idx]
                        f.write("\n".join(hyp_sents))

                    for pos in summary_idx:
                        pos_counts[pos] += 1  # Count index selected
    finally:
        corenlp_tokenizer.stop()

    # Compute evaluation metrics
    if args.compute_rouge:  compute_rouge(args)

    # Position counts
    total_count = sum(pos_counts.values())
    lead_count = pos_counts[0] + pos_counts[1] + pos_counts[2]
    logging.info("Overlap with Lead: {}".format(lead_count / total_count))


def write_rouge_scores(args):
    # Turn off annoying ROUGE logging
    logger = logging.getLogger('global')
    l = logging.StreamHandler(stream=sys.stdout)
    logger.addHandler(l)
    logger.setLevel(logging.WARNING)

    tmp_model_dir = os.path.join(args.result_dir, 'model_one')
    tmp_ref_dir = os.path.join(args.result_dir, 'ref_one')
    temp_dir = os.path.join(args.result_dir, 'temp-files')

    if os.path.exists(tmp_model_dir): rmtree(tmp_model_dir)
    if os.path.exists(tmp_ref_dir): rmtree(tmp_ref_dir)
    os.mkdir(tmp_model_dir)
    os.mkdir(tmp_ref_dir)

    if os.path.exists(temp_dir): rmtree(temp_dir)
    os.mkdir(temp_dir)
    tempfile.tempdir = temp_dir

    signif_stats = {'1': [], '2': [], 'L': [], 'avg': []}

    total_len = len(os.listdir(args.ref_dir))
    for i in tqdm(range(total_len)):
        # Copy file
        model_filename = str(i).rjust(5, '0') + '_hypothesis.txt'
        ref_filename = str(i).rjust(5, '0') + '_reference.txt'
        model_file = os.path.join(args.model_dir, model_filename)
        ref_file = os.path.join(args.ref_dir, ref_filename)
        dest_model_file = os.path.join(tmp_model_dir, model_filename)
        dest_ref_file = os.path.join(tmp_ref_dir, ref_filename)
        copyfile(model_file, dest_model_file)
        copyfile(ref_file, dest_ref_file)

        # Get ROUGE Score
        rouge = Rouge155()
        rouge.system_dir = tmp_model_dir
        rouge.model_dir = tmp_ref_dir
        rouge.system_filename_pattern = '(\d+)_hypothesis.txt'
        rouge.model_filename_pattern = '#ID#_reference.txt'
        output = rouge.convert_and_evaluate()
        output_scores = rouge.output_to_dict(output)

        # Append to score files
        for stat in signif_stats.keys():
            if stat == 'avg':
                output_score = (output_scores['rouge_1_f_score'] + output_scores['rouge_2_f_score'] + output_scores['rouge_l_f_score']) / 3.
            elif stat == 'L':
                output_score = output_scores['rouge_l_f_score']
            else:
                output_score = output_scores['rouge_{}_f_score'.format(stat)]
            signif_stats[stat].append(str(100*output_score))

        # Delete and remake temp dir
        rmtree(temp_dir)
        os.mkdir(temp_dir)
        os.remove(dest_model_file)
        os.remove(dest_ref_file)

    rmtree(tmp_model_dir)
    rmtree(tmp_ref_dir)

    # Write to file
    for stat in signif_stats.keys():
        with open(os.path.join(args.data_dir, "signif_{}.txt".format(stat)), 'w+') as f:
            f.write("\n".join(signif_stats[stat]))


def compute_rouge(args):
    temp_dir = 'temp-files'
    if os.path.exists(temp_dir): rmtree(temp_dir)
    os.mkdir(temp_dir)
    tempfile.tempdir = temp_dir

    logging.info("Computing ROUGE.")
    rouge = Rouge155()
    rouge.system_dir = args.hyp_dir
    rouge.model_dir = args.ref_dir
    rouge.system_filename_pattern = '(\d+)_hypothesis.txt'
    rouge.model_filename_pattern = '#ID#_reference.txt'
    output = rouge.convert_and_evaluate()
    print(output)
    with open(os.path.join(args.data_dir, 'rouge_results.txt'), 'w') as f:
        f.write(output)

    rmtree(temp_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Folder details
    parser.add_argument('--model_file', type=str, default='../model/banditsum_kl_model.pt')
    parser.add_argument('--data_dir', type=str, default='../data/test')
    parser.add_argument('--vocab_file', type=str, default='../data/vocab/vocab_100d.p')

    # Model details
    parser.add_argument('--hidden', type=int, default=200)

    # RL loss details
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--rl_sample_size', type=int, default=20)
    parser.add_argument('--max_num_sents', type=int, default=3)
    parser.add_argument('--kl_method', type=str, default='none', help='none or kl_avg')
    parser.add_argument('--kl_weight', type=float, default=0.0095, help='Ignored if kl_method is none')

    # Testing details
    parser.add_argument('--compute_rouge_only', action='store_true')
    parser.add_argument('--write_rouge_scores', action='store_true')
    args = parser.parse_args()

    # Load vocab file
    logging.info("Opening vocab file: {}".format(args.vocab_file))
    with open(args.vocab_file, 'rb') as f:
        vocab = pickle.load(f, encoding='latin1')

    # Set up configs
    config(args, vocab)

    # Create directories if needed
    args.ref_dir = os.path.join(args.data_dir, 'ref')
    args.hyp_dir = os.path.join(args.data_dir, 'model')
    args.article_dir = os.path.join(args.data_dir, 'articles')
    if not os.path.exists(args.ref_dir): raise ValueError("Please create reference summary directory, named 'ref'.")
    if not os.path.exists(args.article_dir): raise ValueError("Please create article directory, named 'articles'.")
    if not os.path.exists(args.hyp_dir): os.mkdir(args.hyp_dir)

    if args.compute_rouge_only:
        compute_rouge(args)
    elif args.write_rouge_scores:
        args.ref_dir = os.path.join(args.data_dir, 'ref')
        args.model_dir = os.path.join(args.data_dir, 'model')
        write_rouge_scores(args)
    else:
        make_summaries(args)
