"""
Labeled attachment score (LAS):
- percentage of the predicted words that have the same label and head as the reference dependency relationship
Unlabeled attachment score (UAS):
- percentage of predicted words that have the same head as the reference dependency relationship
Label accuracy score:
- percentage of predicted words that have the same label as the reference dependency relationship
"""
import os


def write_metrics_to_file(res, path):
    try:
        with open(path + "conllu_metrics.txt", "w") as writer:
            for key, value in res.items():
                if key == "las" or key == "uas" or key == "label_acc":
                    writer.write(str(key) + " = " + str(value) + " %" + "\n")
                else:
                    writer.write(str(key) + " = " + str(value) + "\n")
    except IOError as e:
        print("({})".format(e))


def compute_conllu_metrics(path, tasks_type):
    """
    Compute las and uas metrics.
    For each deprel and relpos element (tuple -> couple):
    (right label, predicted label)
    """
    res = {"las": 0.0, "uas": 0.0, "label_acc": 0.0, "total": 0, "punct": 0, "<unk>": 0}
    try:
        with open(os.path.join(path, "test_predictions_" + tasks_type[0] + ".txt")) as f1_deprel, \
                open(os.path.join(path, "test_predictions_" + tasks_type[1] + ".txt")) as f2_relpos:
            while True:
                line_file1 = f1_deprel.readline()
                line_file2 = f2_relpos.readline()
                if not line_file1:
                    break
                elements_file1 = line_file1.split(" ")[1::2]  # odd elements  ['(acl|root)', '(det|det)', '(root|obj)', '(flat:name|flat:name)', '(punct|punct)']
                elements_file2 = line_file2.split(" ")[1::2]  # odd elements  ['(2|0)', '(1|1)', '(0|-2)', '(-1|-1)', '(-2|-4)']
                assert len(elements_file1) == len(elements_file2)
                for el in zip(elements_file1, elements_file2):  # ('(acl|root)', '(2|0)')
                    el_deprel = el[0].replace('(', '').replace(')', '').split("|")
                    el_relpos = el[1].replace('(', '').replace(')', '').split("|")
                    if el_deprel[0] == "punct":
                        # count number of "punct" as right label
                        res["punct"] += 1
                    if el_relpos[0] == "<unk>":
                        # count number of "<unk>" as right label
                        res["<unk>"] += 1
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
    except IOError as e:
        print("({})".format(e))
    print("las = ", res["las"], " / ", res["total"])
    print("uas = ", res["uas"], " / ", res["total"])
    print("label_acc = ", res["label_acc"], " / ", res["total"])
    print("punct label = ", res["punct"])
    print("<unk> label = ", res["<unk>"])
    return res


def main():
    res = compute_conllu_metrics(path="results/", tasks_type=["DEPREL", "RELPOS"])
    print("------------------------------------------------------------")
    res["las"] = (res["las"] / res["total"]) * 100
    res["uas"] = (res["uas"] / res["total"]) * 100
    res["label_acc"] = (res["label_acc"] / res["total"]) * 100
    print("las = ", res["las"], "%")
    print("uas (relpos) = ", res["uas"], "%")
    print("label accuracy score (deprel) = ", res["label_acc"], "%")
    write_metrics_to_file(res, path="results/")


if __name__ == "__main__":
    main()
