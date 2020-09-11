"""
Labeled attachment score (LAS):
- è la percentuale delle parole predette che hanno la stessa etichetta e testa della relazione di dipendenza di riferimento
Unlabeled attachment score (UAS):
- è la percentuale delle parole predette che hanno la stessa testa della relazione di dipendenza di riferimento
Label accuracy score:
- è la percentuale delle parole predette che hanno la stessa etichetta della relazione di dipendenza di riferimento (DEPREL-tag, penso)
"""
import os


def compute_conllu_metrics(path, tasks_type):
    las, uas, label_acc, total = 0.0, 0.0, 0.0, 0.0
    punct = 0
    try:
        with open(os.path.join(path, "test_predictions_" + tasks_type[0] + ".txt")) as f1_deprel, \
                open(os.path.join(path, "test_predictions_" + tasks_type[1] + ".txt")) as f2_relpos:
            while True:
                line_file1 = f1_deprel.readline()
                line_file2 = f2_relpos.readline()
                if not line_file1:
                    break
                # print(line_file1.split(" "), line_file2.split(" "))  # ['Evacuata', '(acl|root)', 'la', '(det|det)', 'Tate', '(root|obj)', 'Gallery', '(flat:name|flat:name)', '.', '(punct|punct)', '\n'] ['Evacuata', '(2|0)', 'la', '(1|1)', 'Tate', '(0|-2)', 'Gallery', '(-1|-1)', '.', '(-2|-4)', '\n']
                elements_file1 = line_file1.split(" ")[1::2]  # odd elements  ['(acl|root)', '(det|det)', '(root|obj)', '(flat:name|flat:name)', '(punct|punct)']
                elements_file2 = line_file2.split(" ")[1::2]  # odd elements  ['(2|0)', '(1|1)', '(0|-2)', '(-1|-1)', '(-2|-4)']
                assert len(elements_file1) == len(elements_file2)
                # print(elements_file1)
                # print(elements_file2)
                for el in zip(elements_file1, elements_file2):  # ('(acl|root)', '(2|0)')
                    # print(el)
                    el_deprel = el[0].replace('(', '').replace(')', '').split("|")
                    el_relpos = el[1].replace('(', '').replace(')', '').split("|")
                    # print(el_deprel)  # (acl|root)
                    # print(el_relpos)  # (2|0)
                    if el_deprel[0] == "punct":
                        punct += 1
                    if el_deprel[0] == el_deprel[1] and el_relpos[0] == el_relpos[1]:
                        # same label and head's relative position
                        las += 1
                    if el_deprel[0] == el_deprel[1]:
                        # same label
                        label_acc += 1
                    if el_relpos[0] == el_relpos[1]:
                        # same head's relative position
                        uas += 1
                    total += 1
    except IOError as e:
        print("({})".format(e))
    print("las: ", las, " / ", total)
    print("uas: ", uas, " / ", total)
    print("label_acc: ", label_acc, " / ", total)
    print(punct)
    las_score = (las/total) * 100
    uas_score = (uas / total) * 100
    label_acc_score = (label_acc / total) * 100
    # in total non è contata una frase da 117: 10417 sono in totale di cui 1162 (senza quella da 117) di punct
    return las_score, uas_score, label_acc_score, total


def main():
    las, uas, label_acc_score, total = compute_conllu_metrics(path="results/", tasks_type=["DEPREL", "RELPOS"])
    print("------------------------------------------------------------")
    print("las = ", las, "%")
    print("uas (relpos) = ", uas, "%")
    print("label accuracy score (deprel) = ", label_acc_score, "%")


if __name__ == "__main__":
    main()
