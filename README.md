# Cantonese Instruction Evolver

This project implements the methodologies from [WizardLM: Empowering Large Language Models to Follow Complex Instructions](https://arxiv.org/abs/2304.12244) and [What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning](https://arxiv.org/abs/2312.15685) for generate complex instructions in Cantonese language.  

What is Auto Evol Instruct?
Auto Evol Instruct is a project that aims to evolve instructions for large language models. The project is based on the idea of evolving instructions for large language models to follow complex instructions. The project uses a genetic algorithm to evolve instructions for large language models.

呢個項目應用咗 [WizardLM: Empowering Large Language Models to Follow Complex Instructions](https://arxiv.org/abs/2304.12244) 同埋  [What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning](https://arxiv.org/abs/2312.15685) 嘅方法，嚟生成廣東話嘅複雜指令。

咩係 Auto Evol Instruct？
Auto Evol Instruct 係一個項目，目標係演化大型語言模型嘅指令。呢個項目嘅概念係演化指令，令大型語言模型可以跟從複雜嘅指令。呢個項目用遺傳演算法嚟演化大型語言模型嘅指令。

### Usage


```shell
# DEITA
python deita_evolve.py

# WizardLM
python wizardlm_evolve.py
```

### Citation

```bibtex
@misc{liu2024makesgooddataalignment,
      title={What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning}, 
      author={Wei Liu and Weihao Zeng and Keqing He and Yong Jiang and Junxian He},
      year={2024},
      eprint={2312.15685},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2312.15685}, 
}
```

```bibtex
@misc{xu2023wizardlmempoweringlargelanguage,
      title={WizardLM: Empowering Large Language Models to Follow Complex Instructions}, 
      author={Can Xu and Qingfeng Sun and Kai Zheng and Xiubo Geng and Pu Zhao and Jiazhan Feng and Chongyang Tao and Daxin Jiang},
      year={2023},
      eprint={2304.12244},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2304.12244}, 
}
```