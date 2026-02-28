# Spectral Anatomy of Legislative Networks

A prototype for analyzing the structural "load-bearing" capacity of legislators and votes using SVD Incidence Centrality.

## The Idea

Standard political science measures ideology (DW-NOMINATE).
Standard network science measures centrality (Degree, Betweenness).
**Spectral Anatomy** measures *structural indispensability* using the full spectrum of the voting network.

We use **SVD Incidence Centrality** (Franco et al. 2026), which decomposes the Member-Vote incidence matrix $B$:
$$ C_v(i) = \sum_k \frac{u_{ki}^2}{\sigma_k^2} $$
$$ C_e(j) = \sum_k \frac{v_{kj}^2}{\sigma_k^2} $$

The $1/\sigma^2$ weighting emphasizes the low-frequency modes (Fiedler vector and near-Fiedler modes) that define the global topology (polarization, consensus).

## Prototype Results (116th Congress)

**Top Structural Members:**
1. Justin Amash (I-MI) - The ultimate structural outlier?
2. Collin Peterson (D-MN) - The last conservative Democrat (Bridge?)
3. Matt Gaetz (R-FL) - Partisan enforce?
4. Thomas Massie (R-KY) - "Dr. No"

**Interpretation:**
The metric seems to highlight **unique voting signatures** that define the network's boundaries or bridges. Unlike PageRank (popularity), this measures *spectral contribution*. Amash and Massie often vote against their party (and everyone else), creating unique spectral directions. Peterson bridges the gap.

## Usage

1. Install dependencies: `pip install pandas scipy numpy`
2. Run analysis: `python main.py`
3. Results saved to `member_centrality_116.csv` and `vote_centrality_116.csv`

## Next Steps

1. **Compare with DW-NOMINATE:** Do these central members map to the center (moderates) or the extremes?
2. **Temporal Analysis:** How does the set of "load-bearing" members change from 1990 to 2020?
3. **Hub/Authority Decomposition:** Differentiate between *Agenda Setters* (Hubs) and *Consensus Builders* (Authorities).
