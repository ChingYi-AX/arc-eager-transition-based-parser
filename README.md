# Arc-eager Transition-based Parser with Averaged-Perceptron #


<p align="left">
<a href="https://github.com/huaminghuangtw/<REPO-NAME>"><img src="https://badges.frapsoft.com/os/v3/open-source.svg?v=103" alt="Open Source Love"></a><br/>

### Introduction 
A ArcEager transition-based parser built from scratch, evaluated on English and German tree bank. For feature engineer, there are single-word
features, word pair feature, three-word feature, and distance feature. Besides, morph features specic to German is also used inn the parser.
The parser is modeled by averaged-perceptron. Use Unlabeled Attachment Score(UAS) for evaluation, and get 88.17% on English test set, 82.21 on German test set. 

### Motivation 
  1. No external library, build your won parser step by step ;) 
  2. Make experiments on feature engineer 
  3. The project is orginally the course competition in the course Statisical Dependency Parsing from University Stuttgart. 
  
---

### How to use 
    
Firstly you need to train the model, and then you will get model_english and model_deutsch. Finally, you can evaluate and predict.
You can find data [here](https://drive.google.com/drive/folders/1mIUnGbhp5smGL8Ma41dK8_FOTt28ASlk?usp=sharing).
```
pip install numpy
python3 main.py --model [model/baseline_model] --language [englsih/deutsch] --process [train/test/dev] --file_path [file_path]
    
```    
---

### Performance with all extended feature (15 epochs):

    Unlabeled Attachment Score(UAS) for dev:
    en dev 89.01
    de dev 84.80

    Unlabeled Attachment Score(UAS) for test:
    en test 88.17
    de test 82.21
 
---

### File documentation

    1. utils.py->
        Sentence: store all information for each token in one sentence
        Reader: read sentences from dataset and build them into sentence structure
        Evaluation: evaluate LAS and UAS

    2. parser.py->
        Parsing: check parsing result with oracle parsing during training
        Instance: take transition labels based on features
        State: initialize initial stack, buffer, arcs, left-dependency(ld) and right-dependency(rd)

    3. avg_perceptron.py->
        AvgPerceptron: use feature template and weight from FeatureMapping

    4. feature_engineer.py->
       FeatureMapping: create feature template

    5. main.py->
        Look at *USAGE to see how to use it


---

### Implement parser on train/dev/test phrase:

    Train -> Print the accuracy every 10000 state
    Dev -> print dev accuracy for evaluation
    Test -> The system will compute the result file
    
    
---

### Result Sample 
- German
``` 
    1	Der	der	ART	_	root_morph	2	_	_	_
2	Streit	Streit	NN	_	case=nom|number=sg|gender=masc	6	_	_	_
3	um	um	APPR	_	case=nom|number=sg|gender=masc	2	_	_	_
4	den	der	ART	_	_	5	_	_	_
5	Amtsleiter	Amtsleiter	NN	_	case=acc|number=sg|gender=masc	3	_	_	_
6	fällt	fallen	VVFIN	_	case=acc|number=sg|gender=masc	0	_	_	_
7	mit	mit	APPR	_	number=sg|person=3|tense=pres|mood=ind	6	_	_	_
8	der	der	ART	_	_	9	_	_	_
9	Neugliederung	Neugliederung	NN	_	case=dat|number=sg|gender=fem	7	_	_	_
10	der	der	ART	_	case=dat|number=sg|gender=fem	11	_	_	_
11	Gesundheitsverwaltung	Gesundheitsverwaltung	NN	_	case=gen|number=sg|gender=fem	9	_	_	_
12	in	in	APPR	_	case=gen|number=sg|gender=fem	11	_	_	_
13	Rheinland-Pfalz	Rheinland-Pfalz	NE	_	_	12	_	_	_
14	zusammen	zusammen	PTKVZ	_	case=dat|number=sg|gender=neut	13	_	_	_
15	,	--	$,	_	_	14	_	_	_
16	die	der	PRELS	_	_	25	_	_	_
17	seit	seit	APPR	_	case=nom|number=sg|gender=fem	24	_	_	_
18	dem	der	ART	_	_	20	_	_	_
19	1.	1.	ADJA	_	case=dat|number=sg|gender=masc	20	_	_	_
20	Januar	Januar	NN	_	case=dat|number=sg|gender=masc|degree=pos	17	_	_	_
21	bei	bei	APPR	_	case=dat|number=sg|gender=masc	24	_	_	_
22	den	der	ART	_	_	23	_	_	_
23	Kreisen	Kreis	NN	_	case=dat|number=pl|gender=masc	21	_	_	_
24	angesiedelt	ansiedeln	VVPP	_	case=dat|number=pl|gender=masc	25	_	_	_
25	ist	sein	VAFIN	_	_	24	_	_	_
26	.	--	$.	_	number=sg|person=3|tense=pres|mood=ind	25	_	_	_
``` 
- English 
``` 
    1	The	the	DT	_	_	2	_	_	_
2	move	move	NN	_	_	3	_	_	_
3	stems	stem	VBZ	_	_	0	_	_	_
4	from	from	IN	_	_	3	_	_	_
5	lessons	lesson	NNS	_	_	4	_	_	_
6	learned	learn	VBN	_	_	5	_	_	_
7	in	in	IN	_	_	6	_	_	_
8	Japan	japan	NNP	_	_	7	_	_	_
9	where	where	WRB	_	_	12	_	_	_
10	local	local	JJ	_	_	11	_	_	_
11	competitors	competitor	NNS	_	_	12	_	_	_
12	have	have	VBP	_	_	8	_	_	_
13	had	have	VBD	_	_	12	_	_	_
14	phenomenal	phenomenal	JJ	_	_	15	_	_	_
15	success	success	NN	_	_	13	_	_	_
16	with	with	IN	_	_	15	_	_	_
17	concentrated	concentrated	JJ	_	_	18	_	_	_
18	soapsuds	soapsuds	NNS	_	_	16	_	_	_
19	.	.	.	_	_	3	_	_	_
``` 
    
    
---

### Contact
If you have any question or suggestion, feel free to contact me at haching1105@gmail.com. Contributions are also welcomed. Please open a [pull-request](https://github.com/ChingYi-AX/text-emotion-classification/compare) or an [issue](https://github.com/ChingYi-AX/text-emotion-classification/issues/new) in this repository.



