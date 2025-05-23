<br>
<br>
<br>

<img src="assets/introduction.png">

<div align="center">
  <h1>
  PiFlow: Principle-aware Scientific Discovery with Multi-Agent Collaboration
</h1>
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
    <a href="https://opensource.org/licenses/MIT">
      <img src="https://img.shields.io/badge/License-CC BY NC 4.0-yellow.svg" alt="License: MIT">
    </a>
    &emsp;
    <a href="">
      <img src="https://img.shields.io/badge/AI4SD-Fully Adaptable & Generalizable-blue.svg" alt="License: MIT">
    </a>
    &emsp;
    <a href="https://arxiv.org/abs/2505.15047">
      <img src="https://img.shields.io/badge/arXiv-2505.15047-red.svg" alt="License: MIT">
    </a>
  </p>
</div>


## 👋 Overview
We introduce `PiFlow`, an information-theoretical framework. It uniquely treats automated scientific discovery as a structured uncertainty reduction problem, guided by foundational principles (e.g., scientific laws). This ensures a more systematic and rational exploration of scientific problems.

:ballot_box_with_check: You can directly use our PiFlow for **ANY** of your specific tasks for assisting scientific discovery!


## 📃 Results
PiFlow has demonstrated significant advancements in scientific discovery:
* Evaluated across three distinct scientific domains:
    * 🔬 Nanohelix.
    * 🧬 Bio-molecules.
    * ⚡ Superconductors.
* Markedly improves discovery efficiency, reflected by a **73.55% increase** in the Area Under the Curve (AUC) of property values versus exploration steps.
* Enhances solution quality by an impressive **94.06%** compared to a vanilla agent system.

PiFlow serves as a Plug-and-Play method, establishing a novel paradigm shift in highly efficient automated scientific discovery, paving the way for more robust and accelerated AI-driven research. Our PiFlow accommodates various scenarios (bio-molecules, nanomaterials and superconductors discovery) with experimental conditions (i.e., tools for agent), necessitating little to no prompt engineering for effective agent-level interaction.

## 🔧 Setup and Run

### 1. Launch Dynamic Environment

We have developed three types of experiments named `AgenX...` (e.g., `AgenX_Chembl35`).
To open the `launch.py` for each scenario's task, run:

```shell
python launch.py
````

### 2. Prepare Your API Key

You need to apply for any **OpenAI compatible API KEYs** for calling any models. Ensure the model embedded at the experiment agent is able to use tools.


### 3. Run PiFlow

You can first configure the running commands in the `/configs/` directory, or simply try the demo:

```shell
bash ./run_demo.sh
```


## 🪄 Adapt to Your Own Task

There are many possible areas that can use PiFlow to assist the discovery processes: 

<div style="display: flex; justify-content: space-around;">
  <div style="flex: 1; padding: 5px;">
    <img src="assets/quantums.jpeg" alt="Description of Image 3" style="width: 100%;">
    <p style="text-align: center;"><em>Quantum Materials</em></p>
  </div>
  <div style="flex: 1; padding: 5px;">
    <img src="assets/battery.jpeg" alt="Description of Image 1" style="width: 100%;">
    <p style="text-align: center;"><em>Battery</em></p>
  </div>
  <div style="flex: 1; padding: 5px;">
    <img src="assets/protein.jpeg" alt="Description of Image 2" style="width: 100%;">
    <p style="text-align: center;"><em>Proteins</em></p>
  </div>
</div>

PiFlow offers full flexibility to adapt to your own scenarios, such as quantum science, MOF synthesis, and others. To adapt to your task, you could design tools based on the given examples at `src/tools/`, simply copy one file, e.g., `_nanohelix_tools.py` and create new one named be your scenario (e.g., `_mof_tools.py`). 

### 1. Create your own tools
Replace the tools with your own preferred one (Http tools or others), and finally import the tool into `src/tools/__init__.py`. The PiFlow will automatically find your tools and register it. 

### 2. Configurations
Don't forget to write the configuration file in the `/config/`: For models, just keep it and fill your own API and add tool name to the experiment agent, and for tasks, you should formally define your task similar to the given examples. 

### 3. Test tools and run
Remember to test your tools to make sure it works fine. After that, you could run with command like `run_PiFlow.sh` and ANY prompt engineering are NOT needed for each agent. 


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