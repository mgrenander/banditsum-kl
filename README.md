This repository contains data and code for the EMNLP 2019 paper "[Countering the Effects of Lead Bias in News Summarization](https://www.aclweb.org/anthology/D19-1620/)".
In particular, we include a pretrained model for the KL method and the code to run and evaluate the model.

Please cite this paper if you use our code:

```
@inproceedings{grenander-etal-2019-countering,
    title = "Countering the Effects of Lead Bias in News Summarization via Multi-Stage Training and Auxiliary Losses",
    author = "Grenander, Matt  and
      Dong, Yue  and
      Cheung, Jackie Chi Kit  and
      Louis, Annie",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-1620",
    doi = "10.18653/v1/D19-1620",
    pages = "6019--6024"
}
```

## Data Links
Pre-trained model: <https://drive.google.com/open?id=1-E8IakncMDn5DkSl4hZXbg332ISwpjHG>

Vocab file: <https://drive.google.com/open?id=1QCrb4bpPP7ldpbEthWYRh4hMFOAzTSPP>

Test set data and outputs (needed to evaluate the model): <https://drive.google.com/open?id=171JzaBwLaXFa-vzxj3HUEyY_nVmKEswY>

The folder layout should look like this:

```
banditsum-kl
|--src
|  |-- ..
|--model
|  |-- banditsum_kl_model.pt
|--data
|  |--test
|  |  |--articles
|  |  |  |--000000_article.txt
|  |  |  |--000001_article.txt
|  |  |  ...
|  |  |--ref
|  |  |  |--000000_reference.txt
|  |  |  |--000001_reference.txt
|  |  |  ...
|  |  |--model
|  |  |  |--000000_hypothesis.txt
|  |  |  ...
|  |--vocab
|  |  |--vocab_100d.p
```

## Running the model
Required libraries:
```
pyrouge
torch>=1.3.1
tqdm
stanford-corenlp
numpy
```

Running `python test.py` will start the model evaluation on the test set. The `make_summaries` method in `test.py` gives an example of how to load the model, preprocess raw text and create model predictions.

The results in this repository are _slightly_ different than reported in the paper, due to preprocessing differences.

## Full training code
(Coming soon, link will be posted here)

## Questions?
Feel free to send me an email at matthew dot grenander at mail dot mcgill dot ca.
