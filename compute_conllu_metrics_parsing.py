"""
Example of use:
    python compute_conllu_metrics_parsing.py results/ --task_type DEPREL RELPOS --use-punct

Labeled attachment score (LAS):
- percentage of the predicted words that have the same label and head as the reference dependency relationship
Unlabeled attachment score (UAS):
- percentage of predicted words that have the same head as the reference dependency relationship
Label accuracy score:
- percentage of predicted words that have the same label as the reference dependency relationship
"""
import argparse
import os

# approximate list of symbols (SYM) in the Italian UD. Source: https://universaldependencies.org/it/pos/SYM.html
APPROX_SYM_LIST = ['%', '#', '@', '€', '$', '§', '¢', '©', ':)', ':-)', ':(', ':-(']


def write_metrics_to_file(res, use_punct, path):
    try:
        with open(path + "conllu_metrics.txt", "w") as writer:
            writer.write("Considered punct: " + str(use_punct) + "\n")
            for key, value in res.items():
                if key == "las" or key == "uas" or key == "label_acc":
                    writer.write(str(key) + " = " + str(round(value, 2)) + " %" + "\n")
                else:
                    writer.write(str(key) + " = " + str(round(value, 2)) + "\n")
    except IOError as e:
        print("({})".format(e))


def compute_conllu_metrics(path, tasks_type, use_punct, use_sym):
    """
    :param use_punct: flag to consider punctuation marks when calculating metrics
    :param path: path where there are the results of the model prediction
    :param tasks_type: type of tasks to be considered for the calculation of metrics
    :return: dictionary with metrics

    Compute las and uas metrics.
    For each deprel and relpos element (tuple -> couple):
    (right label, predicted label)
    """
    res = {"las": 0, "uas": 0, "label_acc": 0, "total": 0, "punct": 0, "<unk>": 0, "sym": 0}
    try:
        with open(os.path.join(path, "test_predictions_" + tasks_type[0] + ".txt")) as f1_deprel, \
                open(os.path.join(path, "test_predictions_" + tasks_type[1] + ".txt")) as f2_relpos:
            while True:
                line_file1 = f1_deprel.readline()
                line_file2 = f2_relpos.readline()
                if not line_file1:
                    break
                if use_sym:  # TODO: subtract to the total
                    check_sym(line_file1, res)
                elements_file1 = line_file1.split(" ")[1::2]  # odd elements  ['(acl|root)', '(det|det)', '(root|obj)', '(flat:name|flat:name)', '(punct|punct)']
                elements_file2 = line_file2.split(" ")[1::2]  # odd elements  ['(2|0)', '(1|1)', '(0|-2)', '(-1|-1)', '(-2|-4)']
                assert len(elements_file1) == len(elements_file2)
                for el in zip(elements_file1, elements_file2):  # ('(acl|root)', '(2|0)')
                    el_deprel = el[0].replace('(', '').replace(')', '').split("|")
                    el_relpos = el[1].replace('(', '').replace(')', '').split("|")
                    if el_relpos[0] == "<unk>":
                        # count number of "<unk>" as right label
                        res["<unk>"] += 1
                    if el_deprel[0] != "punct":
                        # get metrics without considered "punct" element
                        calc_metrics(el_deprel, el_relpos, res)
                    else:
                        if use_punct:
                            # count number of "punct" as right label
                            res["punct"] += 1
                            # get metrics considered punct element
                            calc_metrics(el_deprel, el_relpos, res)
                        # else: not call calc_metrics because I don't want considered "punct" label
    except IOError as e:
        print("({})".format(e))
    print("las = ", res["las"], " / ", res["total"])
    print("uas = ", res["uas"], " / ", res["total"])
    print("label_acc = ", res["label_acc"], " / ", res["total"])
    print("punct label = ", res["punct"])
    print("sym = ", res["sym"])
    print("<unk> label = ", res["<unk>"])
    return res


def check_sym(line_file1, res):
    """
    :param line_file1: the file line read
    :param res: the dictionary containing the result
    :return: number of symbols within the line read
    """
    for sym in APPROX_SYM_LIST:
        if sym in line_file1:
            res["sym"] += line_file1.count(sym)


def calc_metrics(el_deprel, el_relpos, res):
    """
    :param el_deprel: deprel tag
    :param el_relpos: relpos tag
    :param res: dictionary with the result
    :return: /
    """
    if el_deprel[0] == el_deprel[1] and el_relpos[0] == el_relpos[1]:
        # same label and head's relative position
        res["las"] += 1
    if el_deprel[0] == el_deprel[1]:
        # same label
        res["label_acc"] += 1
    if el_relpos[0] == el_relpos[1]:
        # same head's relative position
        res["uas"] += 1
    res["total"] += 1


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("results_path", type=str,
                        help="folder path with the results of the model prediction")
    parser.add_argument("--task_type", nargs='+',
                        help="names of the tasks to be evaluated (among those present in tasks.py)")
    parser.add_argument("--use-punct", default=False, action="store_true",
                        help="consider punctuation marks in the evaluation")
    parser.add_argument("--use_sym", default=False, action="store_true",
                        help="consider symbols in the evaluation")
    args = parser.parse_args()
    print(args)

    res = compute_conllu_metrics(args.results_path, args.task_type, args.use_punct, args.use_sym)
    print("------------------------------------------------------------")
    print("Considered punct: ", args.use_punct)
    res["las"] = (res["las"] / res["total"]) * 100
    res["uas"] = (res["uas"] / res["total"]) * 100
    res["label_acc"] = (res["label_acc"] / res["total"]) * 100
    print("las = ", res["las"], "%")
    print("uas (relpos) = ", res["uas"], "%")
    print("label accuracy score (deprel) = ", res["label_acc"], "%")
    write_metrics_to_file(res, args.use_punct, args.results_path)


if __name__ == "__main__":
    main()
