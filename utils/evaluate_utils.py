import numpy as np
import os
import os.path


def average(lst):
    return sum(lst) / float(len(lst))


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


def match_m(all_scores, all_labels):
    """
    This function computes match_m.
    :param all_scores: submission scores
    :param all_labels: ground_truth labels
    :return: match_m dict
    """
    top_m = [1,5,10]
    match_ms = {}
    for m in top_m:
        intersects_lst = []
        # ****************** computing scores:
        score_lst = []
        for s in all_scores:
            # the length of sentence needs to be more than m:
            if len(s) <= m:
                continue
            s = np.array(s)
            ind_score = np.argsort(s)[-m:]
            score_lst.append(ind_score.tolist())
        # ****************** computing labels:
        label_lst = []
        for l in all_labels:
            # the length of sentence needs to be more than m:
            if len(l) <= m:
                continue
            # if label list contains several top values with the same amount we consider them all
            h = m
            if len(l) > h:
                while (h < (len(l)) and l[np.argsort(l)[-h]] == l[np.argsort(l)[-(h + 1)]]):
                    h += 1
            l = np.array(l)
            ind_label = np.argsort(l)[-h:]
            label_lst.append(ind_label.tolist())

        for i in range(len(score_lst)):
            # computing the intersection between scores and ground_truth labels:
            intersect = intersection(score_lst[i], label_lst[i])
            intersects_lst.append((len(intersect))/float((min(m, len(score_lst[i])))))
        # taking average of intersects for the current m:
        match_ms[m] = average(intersects_lst)
    return match_ms

def read_results(filename):
    lines = read_lines(filename) + ['']
    e_freq_lst, e_freq_lsts = [], []
    for line in lines:
        if line:
            splitted = line.split("\t")
            e_freq = splitted[2]
            e_freq_lst.append(float(e_freq))
        elif e_freq_lst:
            e_freq_lsts.append(e_freq_lst)
            e_freq_lst = []
    return e_freq_lsts

def read_labels(filename):
    lines = read_lines(filename) + ['']
    e_freq_lst, e_freq_lsts = [], []
    for line in lines:
        if line:
            splitted = line.split("\t")
            e_freq = splitted[4]
            e_freq_lst.append(float(e_freq))
        elif e_freq_lst:
            e_freq_lsts.append(e_freq_lst)
            e_freq_lst = []
    return e_freq_lsts


def read_lines(filename):
    with open(filename, 'r') as fp:
        lines = [line.strip() for line in fp]
    return lines

# Fix the padding of predicted labels
def fix_padding(scores_numpy, label_probs, mask_numpy):
    """
    Fixes the padding
    :param scores_numpy: predicted scores
    :param label_probs: actual probs
    :param mask_numpy: mask
    :return: scores and labels with no padding
    """
    all_scores_no_padd = []
    all_labels_no_pad = []
    for i in range(len(mask_numpy)):
        all_scores_no_padd.append(scores_numpy[i][:int(mask_numpy[i])])
        all_labels_no_pad.append(label_probs[i][:int(mask_numpy[i])])

    assert len(all_scores_no_padd) == len(all_labels_no_pad)
    return all_scores_no_padd, all_labels_no_pad

if __name__ == '__main__':

    input_dir = os.getcwd()#sys.argv[1]
    output_dir = os.getcwd()#sys.argv[2]

    print("[LOG] input_dir: ", input_dir)
    print("[LOG] output_dir: ", output_dir)

    submit_dir = os.path.join(input_dir, 'res')
    truth_dir  = os.path.join(input_dir, 'ref')

    if not os.path.isdir(submit_dir):
        print("[LOG] {} directory doesn't exist".format(submit_dir))

    if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    output_filename = os.path.join(output_dir, "scores.txt")
    output_file = open(output_filename, "w")

    print("[LOG] Listing files in submission directory: ", os.listdir(submit_dir))
    if not os.path.exists(os.path.join(submit_dir, "submission.txt")):
        print("[LOG] submission.txt file doesn't exist.")
    print("[LOG] reading submission file ...")

    all_score = read_results(os.path.join(submit_dir, "submission.txt"))
    all_label = read_labels(os.path.join(truth_dir, "gold.txt"))

    assert len(all_score) == len(all_label)
    for i in range(len(all_label)):
        assert len(all_label[i]) == len(all_score[i])

    matchm = match_m(all_score, all_label)
    print("[LOG] Match_m: ", matchm)
    print("[LOG] computing RANKING score")

    sum_of_all_scores = 0
    for i, (key,value) in enumerate(matchm.items()):
        output_file.write("score"+str(i+1)+":"+str(value))
        output_file.write("\n")
        sum_of_all_scores+=value
    output_file.write("score:"+str(sum_of_all_scores/float(3))+"\n") #score for final "computed score"
    output_file.close()
