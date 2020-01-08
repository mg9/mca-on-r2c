import json

filepath = 'vcr_val.jsonl'
imgs = []
with open(filepath) as fp:
    img_instance = {}
    line = fp.readline()
    while line:
        img = json.loads(line)
        print("question:", img['question_orig'])
        print("answer:", img['answer_orig'])
        print("reason:", img['rationale_orig'])

        x = img['img_fn']
        image_nm = x.split('/')[len(x.split('/'))-1]
        print("image:", image_nm)

        img_instance = {
            "question" : img['question_orig'] + " [VA] " + img['answer_orig'],
            "answer" : img['rationale_orig'],
            "img_fn" : image_nm
        }
        imgs.append(img_instance)
        line = fp.readline()


with open('r2c_qar.json', 'a') as outfile:
    json.dump(imgs, outfile)
    
