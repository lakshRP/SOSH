# Second Order State Hallucinations (SOSH) for Multi-Agent Systems
---

## 
**Laksh Patel (Illinois Mathematics and Science Academy)**, **Akhilesh Raj (Vanderbilt University)**
## Overview

Second Order State Hallucinations (SOSH) is a novel methodolgy for mitigating attacks in formation control of multi-agent systems. Traditional mulit-agent systems, upon error, experience cascading faults throughout the system. SOSH, utilizing residual analysis, allows each agent to detect faults in the system within a threshold. Then, the network topology is updated to exclude the attacked node(s). Now, as the system lacks the attacked node(s), SOSH comes into action, approximating the attacked node(s) positions with both velocity and acceleration. The depth of approximation (second order) allows for practical use in search-and-rescue, platooning, traffic control, and  military applications.

<div align="center">
  <img src="figures/SOSH.png" alt="Simulation Example" />
</div>
<div align="center">
  Above is an example of SOSH preventing cascading errors on the unaffected nodes (node 1, 3, 4). 
</div>

## 🚀 Getting Started

### Prerequisites

* Python 3.8 or higher
* Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```

### Clone & Navigate

```bash
git clone https://github.com/LakshRP/Priority-Bidding-Mechanism-for-Smart-Intersections.git
cd Priority-Bidding-Mechanism-for-Smart-Intersections
```

### Run Core Experiments

```bash
python code/main.py
```

This produces (in `results/`):

* **Experiment 1: PBM vs. Fixed-Time**

  * `exp1_mean_q_pbm.csv`
  * `exp1_mean_q_fix.csv`
  * `exp1_queue_length.png`

* **Experiment 2: Runtime Scaling**

  * `exp2_vehicle_counts.csv`
  * `exp2_avg_times.csv`
  * `exp2_std_times.csv`
  * `exp2_runtime_scaling.png`

---

## 🔍 Sensitivity Analyses

```bash
python code/analytics.py grid_size      --sizes 2 4 6 8   --nveh 20 --duration 200 --trials 5
python code/analytics.py vehicle_count --grid 4            --counts 10 50 100 200 --duration 200 --trials 5
python code/analytics.py cycle_count   --grid 4 --nveh 20 --cycles 50 100 200 500 --duration 50 --trials 3
```

Outputs saved under `results/analytics/` (CSV summaries and `.png` plots).

---

## 📋 Requirements

```text
numpy>=1.21
matplotlib>=3.4
networkx>=2.6
```



## 📄 License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.


