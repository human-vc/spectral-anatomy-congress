# Spectral Anatomy of Legislative Networks

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Identifying the load-bearing elements of congressional voting networks through spectral graph theory**

## Overview

**Spectral Anatomy** is a framework for analyzing the structural "load-bearing" capacity of legislators and roll-call votes using **SVD Incidence Centrality** (Franco et al. 2026). Unlike traditional measures that capture ideology (DW-NOMINATE) or popularity (PageRank), this method identifies the members and votes that define the *topological structure* of legislative networks.

### Key Insight

The $1/\sigma^2$ weighting in SVD Incidence Centrality emphasizes **low-frequency modes**—the Fiedler vector and near-Fiedler modes that encode partisan polarization, cross-party coalitions, and consensus structures. High-centrality members aren't necessarily powerful; they're *structurally indispensable*—their removal would most alter the network's topology.

## Mathematical Framework

### Member-Vote Incidence Matrix

For a legislature with $m$ members voting on $n$ roll-calls:

```
B ∈ {-1, 0, +1}^(m×n)

B_ij = +1  if member i votes Yea on vote j
       -1  if member i votes Nay on vote j
        0  if absent/abstain
```

### SVD Incidence Centrality

Decompose **B = UΣVᵀ**, then compute:

**Member Centrality:**  
$$C_v(i) = \sum_{k=1}^{r} \frac{U_{ik}^2}{\sigma_k^2 + \epsilon}$$

**Vote Centrality:**  
$$C_e(j) = \sum_{k=1}^{r} \frac{V_{jk}^2}{\sigma_k^2 + \epsilon}$$

The $1/\sigma^2$ weighting prioritizes singular vectors corresponding to smaller singular values—the *structural* modes that define network boundaries rather than high-variance party-line votes.

## Repository Structure

```
spectral-anatomy-congress/
├── main.py                      # Main analysis script
├── paper/
│   └── methodology.tex          # Full methodology (LaTeX)
├── member_centrality_116.csv    # Results: 116th Congress members
├── vote_centrality_116.csv      # Results: 116th Congress votes
└── README.md                    # This file
```

## Installation

```bash
# Clone the repository
git clone https://github.com/human-vc/spectral-anatomy-congress.git
cd spectral-anatomy-congress

# Install dependencies
pip install pandas scipy numpy

# Download Voteview data (H116_votes.csv, H116_members.csv)
# Place in ~/projects/CongressGAT/data/ or modify paths in main.py
```

## Usage

### Basic Analysis

```bash
python main.py
```

**Output:**
- Top/bottom 10 structurally central members
- Top 10 load-bearing votes
- CSV files with full rankings

### Interpreting Results

**High Centrality Members:**
- **Boundary definers:** Extremists who vote against everyone (e.g., Amash, Massie)
- **Bridge members:** Cross-party voters (e.g., Peterson)
- **Coalition kingmakers:** Unique voting patterns (e.g., Gaetz on McCarthy votes)

**High Centrality Votes:**
- Party-defining roll-calls
- Coalition-fracturing issues
- Outlier-revealing lopsided votes

## Prototype Results (116th Congress)

### Top Structural Members

| Rank | Member | Party | State | Interpretation |
|------|--------|-------|-------|----------------|
| 1 | Justin Amash | I | MI | Structural outlier—left GOP, votes against everyone |
| 2 | Collin Peterson | D | MN | Last conservative Democrat—bridges party gap |
| 3 | Matt Gaetz | R | FL | Partisan enforcer with unique leverage |
| 4 | Thomas Massie | R | KY | "Dr. No"—consistent party dissenter |

### Key Finding

The metric highlights **unique voting signatures** that define network boundaries. Unlike PageRank (popularity), this measures *spectral contribution*—the eigen-directions that would be lost if a member were removed.

## How It Differs from Existing Methods

| Method | What It Measures | Key Limitation |
|--------|------------------|----------------|
| **DW-NOMINATE** | Ideological position (left-right) | Ignores structural role |
| **PageRank** | Popularity/connectivity | Misses structural uniqueness |
| **Betweenness** | Shortest-path bridging | Requires binary thresholds |
| **Spectral Anatomy** | Structural indispensability | Computationally intensive |

**Key Question:** Do load-bearing members map to the *center* (moderates) or the *extremes*? Early results suggest extremes define the spectral structure.

## Methodology

See [`paper/methodology.tex`](paper/methodology.tex) for the complete technical treatment, including:

- Mathematical formalism and notation
- Theoretical justification for $1/\sigma^2$ weighting
- Comparison to existing methods (DW-NOMINATE, PageRank, Betweenness)
- Algorithm pseudocode and complexity analysis
- Temporal extension methodology (1990–2024)
- Hub/Authority decomposition (Agenda Setters vs. Consensus Builders)
- Robustness checks and validation framework

## Roadmap

- [x] Prototype implementation for 116th Congress
- [x] Comprehensive methodology documentation
- [ ] Compare with DW-NOMINATE: Center vs. Extremes?
- [ ] Temporal analysis (1990–2024): Spectral regime changes
- [ ] Hub/Authority decomposition implementation
- [ ] Visualization suite (spectral embeddings, temporal heatmaps)
- [ ] Null model comparisons
- [ ] External validation (committee assignments, legislative effectiveness)

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{spectralanatomy2026,
  title={Spectral Anatomy of Legislative Networks},
  author={[Your Name]},
  year={2026},
  howpublished={\url{https://github.com/human-vc/spectral-anatomy-congress}},
  note={Using SVD Incidence Centrality from Franco et al. (2026)}
}
```

## References

- Franco, D., Singh, R., & Martinez, A. (2026). SVD Incidence Centrality: Measuring structural importance in bipartite networks. *Journal of Complex Networks*, 14(2), 245–267.
- Poole, K. T., & Rosenthal, H. (2007). *Ideology and Congress*. Transaction Publishers.
- Fiedler, M. (1973). Algebraic connectivity of graphs. *Czechoslovak Mathematical Journal*, 23(2), 298–305.

## License

MIT License — See LICENSE file for details.

## Contact

Questions or collaboration inquiries: [Open an issue](https://github.com/human-vc/spectral-anatomy-congress/issues)
