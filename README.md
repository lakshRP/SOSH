# Second Order State Hallucinations (SOSH) for Multi-Agent Systems

**Laksh Patel (Illinois Mathematics and Science Academy)**, **Akhilesh Raj (Vanderbilt University)**
## Overview

Second Order State Hallucinations (SOSH) is a novel methodolgy for mitigating attacks in formation control of multi-agent systems. Traditional mulit-agent systems, upon error, experience cascading faults throughout the system. SOSH, utilizing residual analysis, allows each agent to detect faults in the system within a threshold. Then, the network topology is updated to exclude the attacked node(s). Now, as the system lacks the attacked node(s), SOSH comes into action, approximating the attacked node(s) positions with both velocity and acceleration. The depth of approximation (second order) allows for practical use in search-and-rescue, platooning, traffic control, and  military applications.

<div align="center">
  <img src="figures/SOSH" alt="Simulation Example" />
</div>
<div align="center">
  Above is an example of SOSH preventing cascading errors on the unaffected nodes (node 1, 3, 4). 
</div>

## Talks & Awards
- **NCSSS 2025 Student Research Conference**  
  Awarded a fully funded trip to present SOSH at the National Consortium of Secondary STEM Schools.
- **24th Annual High School Research Symposium**  
  Presented SOSH and received the People’s Choice Award.
- **63rd Illinois Junior Science and Humanities Symposium**  
  Presented SOSH research.
- **3rd International Mathematics and Statistics Student Research Symposium**  
  Invited to deliver a talk on SOSH methodology.
  

## 🚀 Getting Started

### Prerequisites

- **Python 3.8+**  
  ```bash
  pip install -r code/requirements.txt


* **MATLAB** (for running `analysis/analysis_extensive.m`)
* **C++17** (with Eigen & matplotlib-cpp for compiling `analysis/analysis_full.cpp`)

### Clone & Navigate

```bash
git clone https://github.com/yourusername/sosh-project.git
cd sosh-project
```

### Generate Dataset

```bash
python analysis/create_csv.py --config analysis/experiment_config.yaml
```

This will produce:

* `results/all_positions.csv`
* `results/aggregated_metrics.csv`

### Analysis

* **MATLAB**
  Open and run `analysis/analysis_extensive.m` to generate figures under `results/figures/`.

* **C++**

  ```bash
  g++ -std=c++17 analysis/analysis_full.cpp -I/path/to/eigen -lpython3.x -o analysis_full
  ./analysis_full
  ```

* **Jupyter Notebook**
  (Optional) Launch `analysis/analysis_notebook.ipynb` for interactive exploration.

### Animation & Visualization

```bash
python code/main.py
```

Runs the SOSH simulation animation with robust detection and second‐order hallucination.

## 📋 Requirements

```text
# code/requirements.txt
numpy>=1.21.0
matplotlib>=3.4.0
pandas>=1.3.0
```

## 📄 License

Released under the **MIT License**. See [LICENSE](LICENSE) for details.


