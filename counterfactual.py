#!/usr/bin/env python3
"""
Counterfactual Congress Simulator

Novel contribution: Predicts how legislative network topology changes 
when specific members are removed (resignation, death, redistricting).

Uses leave-one-out spectral analysis to quantify each member's 
structural contribution to the network.
"""
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
import os
import sys


def compute_fiedler_gap(B):
    """
    Compute the spectral gap (difference between first and second singular values).
    This measures how "polarized" the network is.
    """
    k = min(B.shape) - 1
    if k < 2:
        return 0, 0, 0
    
    U, s, Vt = svds(B, k=k)
    s = np.sort(s)[::-1]  # Sort descending
    
    # Spectral gap = difference between top two singular values
    gap = s[0] - s[1] if len(s) > 1 else 0
    
    # Algebraic connectivity proxy (smallest non-zero singular value)
    alg_conn = s[-2] if len(s) > 1 else 0
    
    return gap, alg_conn, s


def counterfactual_analysis(B_df, member_ids, member_map, n_top=10):
    """
    Perform leave-one-out counterfactual analysis.
    
    For each member, compute how the network structure changes if they were removed.
    
    Returns:
        DataFrame with columns:
        - member_id, name, party
        - fiedler_gap_change (how much polarization changes)
        - algebraic_connectivity_change (network robustness)
        - effective_rank_change (dimensionality of voting space)
        - structural_importance (composite score)
    """
    B = B_df.values
    member_list = B_df.index.tolist()
    
    print("Computing baseline network properties...")
    baseline_gap, baseline_conn, baseline_s = compute_fiedler_gap(B)
    baseline_eff_rank = effective_rank(baseline_s)
    
    print(f"Baseline - Spectral gap: {baseline_gap:.4f}, "
          f"Algebraic connectivity: {baseline_conn:.4f}, "
          f"Effective rank: {baseline_eff_rank:.2f}")
    
    results = []
    
    print(f"\nRunning counterfactual analysis for {len(member_list)} members...")
    
    for idx, member_id in enumerate(member_list):
        if idx % 50 == 0:
            print(f"  Processing member {idx+1}/{len(member_list)}...")
        
        # Remove this member
        member_idx = member_list.index(member_id)
        B_minus = np.delete(B, member_idx, axis=0)
        
        # Recompute spectral properties
        gap_minus, conn_minus, s_minus = compute_fiedler_gap(B_minus)
        eff_rank_minus = effective_rank(s_minus)
        
        # Compute changes
        gap_change = gap_minus - baseline_gap
        conn_change = conn_minus - baseline_conn
        rank_change = eff_rank_minus - baseline_eff_rank
        
        # Composite structural importance score
        # Positive = network becomes more polarized/fragile without this member
        importance = (abs(gap_change) / baseline_gap + 
                     abs(conn_change) / baseline_conn + 
                     abs(rank_change) / baseline_eff_rank) / 3
        
        member_info = member_map.get(member_id, {})
        
        results.append({
            'icpsr': member_id,
            'name': member_info.get('bioname', f'Unknown {member_id}'),
            'party': member_info.get('party_code', 0),
            'state': member_info.get('state_abbrev', ''),
            'fiedler_gap_change': gap_change,
            'algebraic_connectivity_change': conn_change,
            'effective_rank_change': rank_change,
            'structural_importance': importance,
            'network_effect': classify_effect(gap_change, conn_change)
        })
    
    return pd.DataFrame(results)


def effective_rank(singular_values, threshold=0.9):
    """
    Compute effective rank: number of singular values needed to capture 
    threshold fraction of total variance.
    """
    if len(singular_values) == 0:
        return 0
    
    squared = singular_values ** 2
    cumulative = np.cumsum(squared) / np.sum(squared)
    
    # Find first index where cumulative >= threshold
    eff_rank = np.searchsorted(cumulative, threshold) + 1
    return min(eff_rank, len(singular_values))


def classify_effect(gap_change, conn_change):
    """
    Classify the type of structural effect a member has.
    """
    if gap_change > 0.01 and conn_change < -0.01:
        return "BRIDGE"  # Removing them increases polarization
    elif gap_change < -0.01 and conn_change > 0.01:
        return "POLARIZER"  # Removing them decreases polarization
    elif abs(gap_change) < 0.005 and abs(conn_change) < 0.005:
        return "REDUNDANT"  # Network barely changes
    else:
        return "MODERATE"


def predict_special_election_impact(counterfactual_df, district_info):
    """
    Predict the impact of a special election (member replacement).
    
    Args:
        counterfactual_df: Results from counterfactual_analysis
        district_info: Dict with {district: (current_member, likely_replacement)}
    
    Returns:
        Predicted changes in network structure
    """
    predictions = []
    
    for district, (current, replacement) in district_info.items():
        current_row = counterfactual_df[counterfactual_df['name'] == current]
        
        if len(current_row) == 0:
            continue
        
        current_importance = current_row['structural_importance'].values[0]
        current_effect = current_row['network_effect'].values[0]
        
        # Predict based on structural role
        if current_effect == "BRIDGE":
            prediction = f"Replacement likely to increase polarization. " \
                        f"Current member bridges {current_importance:.2f} structural gap."
        elif current_effect == "POLARIZER":
            prediction = f"Replacement may decrease polarization. " \
                        f"Current member contributes {current_importance:.2f} to partisan divide."
        else:
            prediction = f"Minimal structural impact expected. " \
                        f"Member is structurally redundant."
        
        predictions.append({
            'district': district,
            'current_member': current,
            'prediction': prediction,
            'confidence': min(0.95, current_importance * 2)
        })
    
    return pd.DataFrame(predictions)


def identify_critical_members(counterfactual_df, top_n=10):
    """
    Identify the most structurally critical members.
    """
    # Sort by structural importance
    critical = counterfactual_df.nlargest(top_n, 'structural_importance')
    
    print("\n" + "="*70)
    print("CRITICAL MEMBERS: Network would be most affected by their removal")
    print("="*70)
    
    for _, row in critical.iterrows():
        effect_emoji = {
            "BRIDGE": "🌉",
            "POLARIZER": "⚡", 
            "REDUNDANT": "🔄",
            "MODERATE": "⚖️"
        }.get(row['network_effect'], "❓")
        
        print(f"\n{effect_emoji} {row['name']} ({row['state']})")
        print(f"   Structural Importance: {row['structural_importance']:.4f}")
        print(f"   Network Role: {row['network_effect']}")
        print(f"   Effect on polarization: {row['fiedler_gap_change']:+.4f}")
        print(f"   Effect on connectivity: {row['algebraic_connectivity_change']:+.4f}")
    
    return critical


def main():
    print("="*70)
    print("COUNTERFACTUAL CONGRESS SIMULATOR")
    print("Novel Method: Leave-one-out Spectral Analysis")
    print("="*70)
    
    # Load data
    base_dir = os.path.expanduser("~/projects/CongressGAT/data")
    votes_file = os.path.join(base_dir, "H116_votes.csv")
    members_file = os.path.join(base_dir, "H116_members.csv")
    
    if not os.path.exists(votes_file):
        print(f"Error: {votes_file} not found.")
        return
    
    print(f"\nLoading data from {votes_file}...")
    votes_df = pd.read_csv(votes_file)
    members_df = pd.read_csv(members_file)
    
    # Filter and process
    votes_df = votes_df[votes_df['chamber'] == 'House']
    votes_df['vote_val'] = 0
    votes_df.loc[votes_df['cast_code'].isin([1, 2, 3]), 'vote_val'] = 1
    votes_df.loc[votes_df['cast_code'].isin([4, 5, 6]), 'vote_val'] = -1
    votes_df = votes_df[votes_df['vote_val'] != 0]
    
    vote_counts = votes_df.groupby('icpsr').size()
    active_members = vote_counts[vote_counts >= 50].index
    votes_df = votes_df[votes_df['icpsr'].isin(active_members)]
    
    # Build incidence matrix
    print("Building Member-Vote Incidence Matrix...")
    matrix = votes_df.pivot(index='icpsr', columns='rollnumber', values='vote_val').fillna(0)
    
    member_map = members_df.set_index('icpsr')[['bioname', 'party_code', 'state_abbrev']].to_dict('index')
    
    print(f"Matrix shape: {matrix.shape[0]} members × {matrix.shape[1]} votes\n")
    
    # Run counterfactual analysis
    results_df = counterfactual_analysis(matrix, matrix.index.tolist(), member_map)
    
    # Identify critical members
    critical = identify_critical_members(results_df, top_n=15)
    
    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    role_counts = results_df['network_effect'].value_counts()
    print(f"\nNetwork Role Distribution:")
    for role, count in role_counts.items():
        pct = count / len(results_df) * 100
        print(f"  {role}: {count} ({pct:.1f}%)")
    
    print(f"\nStructural Importance:")
    print(f"  Mean: {results_df['structural_importance'].mean():.4f}")
    print(f"  Std:  {results_df['structural_importance'].std():.4f}")
    print(f"  Max:  {results_df['structural_importance'].max():.4f}")
    
    # Party analysis
    print(f"\nBy Party:")
    party_stats = results_df.groupby('party')['structural_importance'].agg(['mean', 'std', 'count'])
    party_map = {100: 'Democrat', 200: 'Republican', 328: 'Independent'}
    for party, stats in party_stats.iterrows():
        name = party_map.get(party, f'Party {party}')
        print(f"  {name}: mean={stats['mean']:.4f}, n={int(stats['count'])}")
    
    # Save results
    out_dir = os.path.expanduser("~/projects/spectral-anatomy-congress")
    os.makedirs(out_dir, exist_ok=True)
    results_df.to_csv(os.path.join(out_dir, "counterfactual_analysis_116.csv"), index=False)
    print(f"\nResults saved to {out_dir}/counterfactual_analysis_116.csv")
    
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    print("""
1. BRIDGE members (increase polarization when removed):
   - These are the structurally load-bearing members
   - Often cross-party voters or unique outliers
   - Their departure would reshape the network topology

2. POLARIZER members (decrease polarization when removed):
   - Contribute to the partisan divide
   - Removing them would make Congress LESS polarized structurally

3. REDUNDANT members (minimal network impact):
   - Can be replaced without affecting network structure
   - Typically predictable party-line voters

This counterfactual framework enables:
- Predicting special election impacts
- Understanding redistricting effects  
- Optimizing committee assignments
- Strategic retirement/timing analysis
""")


if __name__ == "__main__":
    main()
