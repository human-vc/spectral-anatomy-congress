

import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
import os
import sys

def load_data(votes_path, members_path):
    print(f"Loading data from {votes_path}...")
    votes_df = pd.read_csv(votes_path)
    members_df = pd.read_csv(members_path)
    
    
    
    
    
    
    
    
    votes_df = votes_df[votes_df['chamber'] == 'House']
    votes_df['vote_val'] = 0
    votes_df.loc[votes_df['cast_code'].isin([1, 2, 3]), 'vote_val'] = 1
    votes_df.loc[votes_df['cast_code'].isin([4, 5, 6]), 'vote_val'] = -1
    
    
    votes_df = votes_df[votes_df['vote_val'] != 0]

    
    vote_counts = votes_df.groupby('icpsr').size()
    active_members = vote_counts[vote_counts >= 50].index
    votes_df = votes_df[votes_df['icpsr'].isin(active_members)]
    print(f"Filtered to {len(active_members)} active members (50+ votes)")

    
    
    print("Building Incidence Matrix (Member x Vote)...")
    matrix = votes_df.pivot(index='icpsr', columns='rollnumber', values='vote_val').fillna(0)
    
    
    member_map = members_df.set_index('icpsr')[['bioname', 'party_code', 'state_abbrev', 'nominate_dim1']].to_dict('index')
    
    return matrix, member_map

def compute_svd_centrality(B_df, regularization=0.99):
    
    B = B_df.values
    n_members, n_votes = B.shape
    print(f"Matrix shape: {n_members} Members x {n_votes} Votes")
    
    
    
    
    
    
    
    
    print("Computing SVD...")
    
    k = min(n_members, n_votes, 50) - 1
    if k < 1: k = 1
    U, s, Vt = svds(B, k=k)
    
    
    idx = np.argsort(s)[::-1]
    U = U[:, idx]
    s = s[idx]
    Vt = Vt[idx, :]
    
    print(f"Top 5 Singular Values: {s[:5]}")
    
    
    weights = 1.0 / (s**2)
    
    
    
    Cv = np.sum(U**2 * weights, axis=1)
    
    
    
    V = Vt.T
    Ce = np.sum(V**2 * weights, axis=1)
    
    return Cv, Ce

def analyze_results(Cv, Ce, matrix, member_map):
    
    members = matrix.index
    member_scores = []
    for i, icpsr in enumerate(members):
        info = member_map.get(icpsr, {'bioname': f'Unknown {icpsr}', 'party_code': 0})
        member_scores.append({
            'icpsr': icpsr,
            'name': info['bioname'],
            'party': info['party_code'], 
            'state': info.get('state_abbrev', ''),
            'centrality': Cv[i]
        })
    
    mem_df = pd.DataFrame(member_scores)
    mem_df = mem_df.sort_values('centrality', ascending=False)
    
    
    votes = matrix.columns
    vote_scores = []
    for i, roll in enumerate(votes):
        vote_scores.append({
            'rollnumber': roll,
            'centrality': Ce[i]
        })
    
    vote_df = pd.DataFrame(vote_scores)
    vote_df = vote_df.sort_values('centrality', ascending=False)
    
    return mem_df, vote_df

def main():
    base_dir = os.path.expanduser("~/projects/CongressGAT/data")
    votes_file = os.path.join(base_dir, "H116_votes.csv")
    members_file = os.path.join(base_dir, "H116_members.csv")
    
    if not os.path.exists(votes_file):
        print(f"Error: {votes_file} not found.")
        return

    B_df, member_map = load_data(votes_file, members_file)
    Cv, Ce = compute_svd_centrality(B_df)
    mem_df, vote_df = analyze_results(Cv, Ce, B_df, member_map)
    
    print("\n" + "="*60)
    print("SPECTRAL ANATOMY: TOP 10 CENTRAL MEMBERS (116th Congress)")
    print("Interpretation: Members most embedded in the low-frequency consensus/conflict structure")
    print("="*60)
    print(mem_df.head(10)[['name', 'party', 'state', 'centrality']].to_string(index=False))
    
    print("\n" + "="*60)
    print("SPECTRAL ANATOMY: BOTTOM 10 CENTRAL MEMBERS (Outliers/Mavericks)")
    print("="*60)
    print(mem_df.tail(10)[['name', 'party', 'state', 'centrality']].to_string(index=False))

    print("\n" + "="*60)
    print("SPECTRAL ANATOMY: TOP 10 LOAD-BEARING VOTES")
    print("Interpretation: Votes that define the structural alignment of the Congress")
    print("="*60)
    print(vote_df.head(10).to_string(index=False))

    
    out_dir = os.path.expanduser("~/projects/spectral-anatomy-congress")
    mem_df.to_csv(os.path.join(out_dir, "member_centrality_116.csv"), index=False)
    vote_df.to_csv(os.path.join(out_dir, "vote_centrality_116.csv"), index=False)
    print(f"\nResults saved to {out_dir}")

if __name__ == "__main__":
    main()
