

# 🙃 Scare Quotes as Markers of "Questionable" Word Usages 👻

This repository contains the annotations, guidelines and code related to the upcoming paper:

Aina Garí Soler, Juan Carlos Zevallos Huaco, Matthieu Labeau and Chloé Clavel (2026). Scare Quotes as Markers of “Questionable” Word Usages and Misalignment in Conversation: An Annotation Study. Accepted to the Fifteenth Language Resources and Evaluation Conference: LREC 2026, Palma de Mallorca, Spain, May 13-15.

All code used for experiments in the paper was first written manually; then it was debugged and its presentation was improved using Claude 4.6.



## Obtaining SQUiC annotations and guidelines

Visit the ``obtaining_data/`` directory for instructions on how to obtain our full annotations with text.

`annotation_guidelines/` contains the annotation guidelines (the ones used for the paper and the subsequently updated version).


## Running annotation analyses


### Getting annotation statistics

To obtain the statistics presented in Sections 4.1 and 4.2 of the paper, run the analyses in the notebook `getting_annotation_statistics.ipynb`.

To run the complete analysis (specifically, those on instances involving Word Meaning Negotiation), you will need a copy of the NeWMe corpus. Visit [this repository](https://github.com/ainagari/WMNindicator_detection/tree/main/obtaining_data).

For the lexical analyses of quoted unigrams presented in Section 4.3, run the notebook `quoted_passages_analysis.ipynb`. Information on how to download the relevant lexica is found in the notebook header.


### Calculating Inter-Annotator Agreement

The code used for calculating the inter-annotator agreement is found in the notebook `IAA.ipynb`.


## LLM annotation experiments

You can run `llm_calls.py` to make predictions with LLMs. Indicate the model you want to use using the ``--model`` flag (``olmo``, ``qwen``, ``llama70b``). For Llama 70B, you need to indicate your token and path to the model in the script. Indicate also the subset of the data where you want to run predictions (``--subset [all|iaa|dev|test]``) and whether you want to make predictions with all prompts or only the ones that showed best results on the dev set for each model (``--prompt_types [all|best]``).

This saves model predictions in {subset}\_results.

The script ``complete_test_evaluation.py`` evaluates the model predictions.

To calculate the inter-annotator agreement results treating LLMs as second annotators, go to the section "Agreement with LLMs" in ``IAA.ipynb`` (adjust filenames if necessary).



### Citation

If you use our corpus, please cite our paper:

```
Citation upcoming

```


### Contact

For any questions or requests feel free to [contact me](https://ainagari.github.io/menu/contact.html).

