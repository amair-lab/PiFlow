

<div align="center">
  <h1>PiFlow: Principle-aware Scientific Discovery with Multi-Agent Collaboration</h1>
</div>


<div align="center">

[Mellen Y. Pu](https://dandelionym.github.io/)
&emsp;&emsp;&emsp;
[Tao Lin]()
&emsp;&emsp;&emsp;
[Hongyu Chen]()
 
Westlake University

</div>

<div align="center">
  <p>
    <a href="https://creativecommons.org/licenses/by-nc/4.0/">
      <img src="https://img.shields.io/badge/License-CC BY NC 4.0-yellow.svg" alt="License: CC BY NC 4.0">
    </a>
    &emsp;
    <a href="">
      <img src="https://img.shields.io/badge/AI4SD-Fully Adaptable & Generalizable-blue.svg" alt="AI4SD Fully Adaptable & Generalizable">
    </a>
    &emsp;
    <a href="https://arxiv.org/pdf/2505.15047v2">
      <img src="https://img.shields.io/badge/arXiv-2505.15047-red.svg" alt="arXiv: 2505.15047">
    </a>
  </p>
</div>


> **New to PiFlow?** <img src="assets/colab-icon.png" width="24px" style="vertical-align: middle;"> Check out our [Colab Tutorial](tutorials/PiFlow_Tutorial.ipynb) to get started in minutes! 


## 👋 Overview

We introduce `PiFlow` (Principle Flow), an information-theoretical framework that treats automated scientific discovery as a structured uncertainty reduction problem, guided by foundational scientific principles. This approach ensures a more systematic and rational exploration of scientific problems by learning and optimization. You can directly use PiFlow for **ANY** of your specific tasks to assist scientific discovery!

<br/>

<div align="center">
  <img src="assets/introduction.png" alt="PiFlow Introduction">
</div>

<br/>

🔮 Supported Features:

- Budget-limited iterative hypothesis-testing.
- Saving the full running log of agents.
- Colored output of different agents in terminal.
- [x] **Interactive Tutorial**: Step-by-step guidance for scientists.
- Case study:
  - Discovered nanohelix geo-structure with high chiral property (g-factor > 1.8).

## 📃 Primary Results

PiFlow has demonstrated significant advancements in scientific discovery across multiple domains:

* **Comprehensive evaluation**: Tested across three distinct scientific domains:
    * 🔬 Nanohelix materials
    * 🧬 Bio-molecules  
    * ⚡ Superconductors

* **Enhanced efficiency**: Demonstrates a **73.55% increase** in the Area Under the Curve (AUC) of property values versus exploration steps, significantly improving discovery efficiency.

* **Superior solution quality**: Achieves an impressive **94.06% improvement** in solution quality compared to a baseline agent system.

<div align="center">
  <img src="assets/results.png" alt="PiFlow Experimental Results" width="80%">
</div>

PiFlow serves as a plug-and-play method that establishes a novel paradigm for highly efficient automated scientific discovery, paving the way for more robust and accelerated AI-driven research. The framework accommodates various scenarios (bio-molecules, nanomaterials, and superconductor discovery) with diverse experimental conditions, requiring minimal to no prompt engineering for effective agent-level interaction.

## 🔧 Setup and How-to-Run

### 0. Installation

To prepare the environment, execute the following conda commands: 

```shell
git clone https://github.com/amair-lab/PiFlow && cd PiFlow
conda env create -f environment.yml  # The environment name will be `piflow`
conda activate piflow 

# We recommend using mamba for faster installation, though conda works as well
# If you want to use mamba for package management, run these commands instead:
# conda install mamba -n base -c conda-forge
# mamba env create -f environment.yml
# mamba activate piflow
```

### 1. Configuration

You need to obtain the **OpenAI-compatible API key** for calling the language models. Ensure that the model embedded within the experimental agent is able to use tools. We recommend using official large models supported by [Alibaba Cloud](https://www.alibabacloud.com/help/zh/model-studio/models), e.g. QwenMax.

In the configuration for language models, each agent's profile contains API info and settings like verbose mode:

```yaml
streaming:  # bool, true means the output will be in streaming format, consistent with the original API output
api_config:
    base_url:       # Follow the OpenAI API format
    model_name:     # Follow the OpenAI API format
    is_reasoning:   # bool, true if it is a reasoning model like o1, R1, QwQ, etc. (typically has longer processing time)
    api_key:        # Follow the OpenAI API format
    temperature:    # Follow the OpenAI API, generally set to 0.6 or 0.4
    max_tokens:     # We set this to 4096
tools: []           # A list of tool names; they should be defined and declared in `tools/__init__.py`, following YAML list syntax
```

Other fields like `enabled` indicate whether this agent is a member of group chat. Fields like `name` and `description` are used for building the agent profile. Note that the `name` field should not contain any blank characters.

After configuration, you can run the demo task for nanohelix material discovery (you could also modify the args by `argparse`, see `inference.py`) by running the command:

```shell
python inference.py
```

You will see detailed output with colored chat history at the command line if success. Our philosophy is that **one configuration file corresponds to one task**, and you should prepare both task and model configurations for running PiFlow.

## 🪄 Try PiFlow!

PiFlow also offers extensive flexibility to adapt to your own scenarios, such as quantum science, MOF synthesis, and other domains, as demonstrated in our manuscript - its **Plug-and-Play** ability. The framework can assist with various discovery processes.

To adapt PiFlow to your task, you can design tools based on the provided examples in `tools/...`. Simply copy one file (e.g., `[tool_nanomaterial.py](tools/tool_nanomaterial.py)_nanohelix_tools.py`) and create a new file named according to your scenario (e.g., `_mof_tools.py`), the tools manager will automatically load this new tool, but please remember to add `@register_tool` for assigning both name and desc. Additionally, you should also prepare the output as a JSON format data with the same fields supported by us:

```python
{
    "tool_name": "xxx", # same with `@register_tool`
    "input": "",    # exact hypothesis candidate, str, etc
    "output": None,     # the `reward` for PiFlow to maximize
    "success": True,  # boolean value for checking the status
    "error": "xxx"  # helper info
}
```

Remember to add the tool into the agent profile at model's config file. This is essential because the program will screen all tools defined at `tools/` and load them into the agent's context by filtering the tool name.  

## 📚 Citation

```bibtex
@misc{pu2025piflow,
      title={PiFlow: Principle-aware Scientific Discovery with Multi-Agent Collaboration}, 
      author={Yingming Pu and Tao Lin and Hongyu Chen},
      year={2025},
      eprint={2505.15047},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.15047}, 
}
```

## 📄 License

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License](https://creativecommons.org/licenses/by-nc/4.0/). Under this license, you are free to share and adapt this work for non-commercial purposes, provided you give appropriate attribution.