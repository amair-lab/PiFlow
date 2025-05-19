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
    <a href="https://opensource.org/licenses/MIT">
      <img src="https://img.shields.io/badge/AI4SD-Fully Adaptable & Generalizable-blue.svg" alt="License: MIT">
    </a>
  </p>
</div>

## 👋 Overview
We introduce `PiFlow`, an information-theoretical framework. It uniquely treats automated scientific discovery as a structured uncertainty reduction problem, guided by foundational principles (e.g., scientific laws). This ensures a more systematic and rational exploration of scientific problems.

:ballot_box_with_check: You can directly use our PiFlow for **ANY** of your specific tasks for assisting scientific discovery!


## 📃 Results
PiFlow has demonstrated significant advancements in scientific discovery:
* Evaluated across three distinct scientific domains:
    * 🔬 Discovering nanomaterial structures.
    * 🧬 Bio-molecules.
    * ⚡ Superconductor candidates with targeted properties.
* Markedly improves discovery efficiency, reflected by a **73.55% increase** in the Area Under the Curve (AUC) of property values versus exploration steps.
* Enhances solution quality by an impressive **94.06%** compared to a vanilla agent system.

PiFlow serves as a Plug-and-Play method, establishing a novel paradigm shift in highly efficient automated scientific discovery, paving the way for more robust and accelerated AI-driven research. Our PiFlow accommodates various scenarios (bio-molecules, nanomaterials and superconductors discovery) with experimental conditions (i.e., tools for agent), necessitating little to no prompt engineering for effective agent-level interaction.

## 🔧 Setup

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

## 📚 Citation
```bibtex

```


## 📄 License
This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License](https://creativecommons.org/licenses/by-nc/4.0/). Under this license, you are free to share and adapt this work for non-commercial purposes, provided you give appropriate attribution.