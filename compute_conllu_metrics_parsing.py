"""
Example of use:
    python compute_conllu_metrics_parsing.py results/ --task_type DEPREL RELPOS --consider_punct --consider_sym
    --details

Labeled attachment score (LAS):
- percentage of the predicted words that have the same label and head as the reference dependency relationship
Unlabeled attachment score (UAS):
- percentage of predicted words that have the same head as the reference dependency relationship
Label accuracy score:
- percentage of predicted words that have the same label as the reference dependency relationship
"""
import argparse
import os
import logging

# approximate list of symbols (SYM) in the Italian UD. Source: https://universaldependencies.org/it/pos/SYM.html
APPROX_SYM_LIST = ['%', '#', '@', '€', '$', '§', '¢', '©', ':)', ':-)', ':(', ':-(']

logger = logging.getLogger(__name__)


def write_metrics_to_file(metrics_dict, args):
    try:
        with open(args.results_path + "conllu_metrics.txt", "w") as writer:
            writer.write("Considered punct: " + str(args.consider_punct) + "\n")
            writer.write("Considered sym: " + str(args.consider_sym) + "\n")
            for key, value in metrics_dict.items():
                if key == "las" or key == "uas" or key == "label_acc":
                    writer.write(str(key) + " = " + str(round(value, 2)) + " %" + "\n")
                else:
                    writer.write(str(key) + " = " + str(round(value, 2)) + "\n")
    except IOError as e:
        print("({})".format(e))


def compute_conllu_metrics(args):
    """
    :return: dictionary with metrics

    Compute las and uas metrics.
    For each deprel and relpos element (tuple -> couple):
    (right label, predicted label)
    """
    metrics_dict = {"las": 0, "uas": 0, "label_acc": 0, "total": 0, "punct": 0, "sym": 0, "<unk>": 0}
    try:
        with open(os.path.join(args.results_path, args.set + "_predictions_" + args.task_type[0] + ".txt")) as f1_deprel, \
                open(os.path.join(args.results_path, args.set + "_predictions_" + args.task_type[1] + ".txt")) as f2_relpos:
            while True:
                line_deprel_file = f1_deprel.readline()
                line_relpos_file = f2_relpos.readline()
                if not line_deprel_file:
                    break
                flag, sym = check_sym(line_deprel_file, metrics_dict)
                if not args.consider_sym and flag:
                    line_deprel_file, line_relpos_file = remove_sym(line_deprel_file, line_relpos_file, sym)
                    metrics_dict["sym"] = 0
                deprel_elements = line_deprel_file.split(" ")[1::2]  # odd elements  ['(acl|root)', '(det|det)', '(root|obj)', '(flat:name|flat:name)', '(punct|punct)']
                relpos_elements = line_relpos_file.split(" ")[1::2]  # odd elements  ['(2|0)', '(1|1)', '(0|-2)', '(-1|-1)', '(-2|-4)']
                assert len(deprel_elements) == len(relpos_elements)
                for el in zip(deprel_elements, relpos_elements):  # ('(acl|root)', '(2|0)')
                    el_deprel = el[0].replace('(', '').replace(')', '').split("|")
                    el_relpos = el[1].replace('(', '').replace(')', '').split("|")
                    if el_relpos[0] == "<unk>":
                        # count number of "<unk>" as right label
                        metrics_dict["<unk>"] += 1
                    if el_deprel[0] != "punct":
                        # get metrics without considered "punct" element
                        calc_metrics(el_deprel, el_relpos, metrics_dict)
                    else:
                        if args.consider_punct:
                            # count number of "punct" as right label
                            metrics_dict["punct"] += 1
                            # get metrics considered punct element
                            calc_metrics(el_deprel, el_relpos, metrics_dict)
                        # else: not call calc_metrics because I don't want considered "punct" label
    except IOError as e:
        print("({})".format(e))
    if args.details:
        detailed_print(metrics_dict)
    return metrics_dict


def detailed_print(metrics_dict):
    logger.info("las = %s / %s ", metrics_dict["las"], metrics_dict["total"])
    logger.info("uas = %s / %s ", metrics_dict["uas"], metrics_dict["total"])
    logger.info("label_acc = %s / %s ", metrics_dict["label_acc"], metrics_dict["total"])
    logger.info("punct label = %s ", metrics_dict["punct"])
    logger.info("sym = %s ", metrics_dict["sym"])
    logger.info("<unk> label = %s ", metrics_dict["<unk>"])


def remove_sym(line_deprel_file, line_relpos_file, sym):
    deprel_list = line_deprel_file.split(" ")
    relpos_list = line_relpos_file.split(" ")
    # get position of all SYM occurrences in tmp_file1
    indices = [i for i, x in enumerate(deprel_list) if sym in x]
    for i in range(0, len(indices)):
        # given indices remove SYM from lists
        res_deprel = deprel_list[:indices[i] - (2 * i)] + deprel_list[indices[i] - (2 * i) + 2:]
        res_relpos = relpos_list[:indices[i] - (2 * i)] + relpos_list[indices[i] - (2 * i) + 2:]
        assert len(res_deprel) == len(res_relpos)
        deprel_list = res_deprel
        relpos_list = res_relpos
    # create a new string without SYM
    new_deprel_line = ' '.join(res_deprel)
    new_relpos_line = ' '.join(res_relpos)
    assert new_deprel_line.count(sym) == 0 and new_relpos_line.count(sym) == 0
    return new_deprel_line, new_relpos_line


def check_sym(line_deprel_file, metrics_dict):
    """
    :param line_deprel_file: file line read
    :param metrics_dict: dict with results
    :return: True and symbol if there is at least one symbol, False and empty string otherwise
    """
    for sym in APPROX_SYM_LIST:
        if sym in line_deprel_file:
            metrics_dict["sym"] += line_deprel_file.count(sym)
            return True, sym
        else:
            return False, ''


def calc_metrics(el_deprel, el_relpos, metrics_dict):
    """
    :param el_deprel: deprel tag
    :param el_relpos: relpos tag
    :param metrics_dict: dictionary with the result
    :return: /
    """
    if el_deprel[0] == el_deprel[1] and el_relpos[0] == el_relpos[1]:
        # same label and head's relative position
        metrics_dict["las"] += 1
    if el_deprel[0] == el_deprel[1]:
        # same label
        metrics_dict["label_acc"] += 1
    if el_relpos[0] == el_relpos[1]:
        # same head's relative position
        metrics_dict["uas"] += 1
    metrics_dict["total"] += 1


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("results_path", type=str,
                        help="folder path with the results of the model prediction")
    parser.add_argument("--task_type", nargs='+',
                        help="list of names of the tasks to be evaluated (among those present in tasks.py)")
    parser.add_argument("--consider_punct", default=False, action="store_true",
                        help="consider punctuation marks in the evaluation")
    parser.add_argument("--consider_sym", default=False, action="store_true",
                        help="consider symbols in the evaluation")
    parser.add_argument("--details", default=False, action="store_true",
                        help="to print detailed results")
    parser.add_argument("--set", type=str, default="test",
                        help="set the set type (e.g. test set or dev set)")
    args = parser.parse_args()
    print(args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    metrics_dict = compute_conllu_metrics(args)
    logger.info("-"*50)
    logger.info("Considered punct: %s ", args.consider_punct)
    logger.info("Considered sym: %s ", args.consider_sym)

    metrics_dict["las"] = (metrics_dict["las"] / metrics_dict["total"]) * 100
    metrics_dict["uas"] = (metrics_dict["uas"] / metrics_dict["total"]) * 100
    metrics_dict["label_acc"] = (metrics_dict["label_acc"] / metrics_dict["total"]) * 100

    logger.info("las = %s %s", metrics_dict["las"], "%")
    logger.info("uas (relpos) = %s %s", metrics_dict["uas"], "%")
    logger.info("label accuracy score (deprel) = %s %s", metrics_dict["label_acc"], "%")

    write_metrics_to_file(metrics_dict, args)


if __name__ == "__main__":
    main()
