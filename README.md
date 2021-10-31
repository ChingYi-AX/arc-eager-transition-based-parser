# Arc-eager Transition-based Parser with Averaged-Perceptron #

*Statistical Dependency Parsing project in IMS, Uni-Stuttgart*

*Ching-Yi Chen, 22.Feb.2021*


*Code documentation:

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


*USAGE:

    python3 main.py --model [model/baseline_model] --language [englsih/deutsch] --process [train/test/dev] --file_path [file_path]
    ***Firstly you need to train the model, and then you will get model_english and model_deutsch. Finally, you can evaluate and predict.

*How to implement the Arc-eager transition-based parser?

    Train -> Print the accuracy every 10000 state
    Dev -> print dev accuracy for evaluation
    Test -> The system will compute the result file


*Note:

    baseline model dose not included in this package, baseline model is only used for experiments
    some extended feature only used for experiment did not include in this package (e.g. morph feature)


*Performance with extended feature (15 epochs):

    Result for dev:
    en dev 89.01
    de dev 84.80

    Result for test:
    en test 88.17
    de test 82.21


