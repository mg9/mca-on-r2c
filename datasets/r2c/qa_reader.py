import json

train = ('datasets/r2c/train.jsonl', 'datasets/r2c/r2c_qar_test.json')
test =  ('datasets/r2c/test.jsonl', 'datasets/r2c/r2c_qar_train.json')

dsets = [train, test]

for dset in dsets:
    with open(dset[0]) as fp:
        instances = []
        line = fp.readline()
        while line:
            img = json.loads(line)
            rat_choices = img['rationale_choices']
            rat_label = img['rationale_label']

            for idx, i in enumerate(rat_choices):
                if idx != rat_label:
                    answer = ' '.join(map(str, i)).replace('[', '').replace(']','')
                    nonreason_instance = {
                    "question" : img['question_orig'] + " [VA] " + img['answer_orig'],
                    "answer" : answer ,
                    "img_fn" : img['img_fn'],
                    "metadata_fn": img['metadata_fn'],
                    "objects": img['objects'],
                    "label" : 0
                    }
                    instances.append(nonreason_instance)

            img_instance = {
                "question" : img['question_orig'] + " [VA] " + img['answer_orig'],
                "answer" : img['rationale_orig'],
                "img_fn" : img['img_fn'],
                "metadata_fn": img['metadata_fn'],
                "objects": img['objects'],
                "label": 1
            }
            instances.append(img_instance)
            line = fp.readline()


    with open(dset[1], 'a') as outfile:
        json.dump(instances, outfile)

