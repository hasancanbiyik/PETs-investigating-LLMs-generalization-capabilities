# PETs: Investigating LLMs’ Generalization Capabilities

This repository contains the code, data splits, and experimental scripts for the paper:

**When Semantic Overlap Is Not Enough: Cross-Lingual Euphemism Transfer Between Turkish and English**

The project investigates cross-lingual generalization in euphemism detection by categorizing Potentially Euphemistic Terms (PETs) into **Overlapping (OPETs)** and **Non-Overlapping (NOPETs)** subsets for English and Turkish.

---

## Repository Structure

```
.
├── baseline/
│   └── baseline.ipynb
├── datasets/
│   ├── EN_OPETs.csv
│   ├── EN_NOPETs_fixed.csv
│   ├── TR_OPETs.csv
│   └── TR_NOPETs_fixed.csv
├── en_train_xlmr_OPETs_splits/
├── en_train_xlmr_NOPETs_splits/
├── tr_train_xlmr_OPETs_splits/
├── tr_train_xlmr_NOPETs_splits/
├── zeroshot_experiment/
│   ├── run_gpt4_experiment.py
│   ├── analyze_category_distributions.py
│   └── gpt4o_bias_table.tex
└── README.md
```

---

## Contents

- **datasets/**  
  English and Turkish PET datasets, split into OPET and NOPET subsets, used in all experiments.

- **baseline/**  
  Baseline experiments using frozen XLM-R representations with a logistic regression classifier.

- **\*_train_xlmr_\*_splits/**  
  Train/validation/test splits used for fine-tuning XLM-R on OPET and NOPET subsets in each language.

- **zeroshot_experiment/**  
  Scripts for zero-shot evaluation with GPT-4o, including inference, category analysis, and table generation.

---

## Experiments

This repository supports:
- Baseline evaluation with frozen multilingual embeddings
- Fine-tuning XLM-R on OPET and NOPET subsets
- Zero-shot euphemism classification using GPT-4o with language-matched prompts

Experimental details and prompt templates are described in the paper and Appendix B.

---

## Citation

If you use this repository, please cite the corresponding paper.

---

## License

This repository is released for research purposes.
