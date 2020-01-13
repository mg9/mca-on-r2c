import json

filepath = '/Users/mugekural/mca-on-r2c/datasets/r2c/vcr_test.jsonl'
imgs = []
nonreasons = []
with open(filepath) as fp:
    img_instance = {}
    line = fp.readline()
    while line:
        img = json.loads(line)
        #print("question:", img['question_orig'])
        #print("answer:", img['answer_orig'])
        #print("reason:", img['rationale_orig'])
        #print("label:", 1)

        rat_choices = img['rationale_choices']
        rat_label = img['rationale_label']


        x = img['img_fn']
        image_nm = x.split('/')[len(x.split('/'))-1]
        #print("image:", image_nm)


        for idx, i in enumerate(rat_choices):
            if idx != rat_label:
                nonreason_instance = {
                "question" : img['question_orig'] + " [VA] " + img['answer_orig'],
                "answer" : ' '.join(map(str, i)) ,
                "img_fn" : image_nm,
                "label" : 0
                }
                print(nonreason_instance)
                nonreasons.append(nonreason_instance)

        img_instance = {
            "question" : img['question_orig'] + " [VA] " + img['answer_orig'],
            "answer" : img['rationale_orig'],
            "img_fn" : image_nm,
            "label": 1
        }
        imgs.append(img_instance)
        line = fp.readline()


with open('r2c_qar_test.json', 'a') as outfile:
    json.dump(imgs, outfile)
    json.dump(nonreasons, outfile)

