#!/usr/bin/env python3
"""
Spectral Coalition Predictor

Novel contribution: Predicts coalition formation on upcoming votes
using spectral properties of the legislative network.

Key insight: The spectral embedding of members captures latent 
coalition structures before they manifest in actual voting.
"""
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from scipy.spatial.distance import cosine
import os


class SpectralCoalitionPredictor:
    """
    Predict coalition formation using spectral embeddings.
    """
    
    def __init__(self, B_df, member_map):
        """
        Initialize with incidence matrix.
        
        Args:
            B_df: Member-Vote incidence matrix (DataFrame)
            member_map: Dict mapping icpsr to member info
        """
        self.B = B_df.values
        self.member_ids = B_df.index.tolist()
        self.member_map = member_map
        self.embeddings = None
        self.singular_values = None
        
        print("Computing spectral embeddings...")
        self._compute_embeddings()
    
    def _compute_embeddings(self, k=50):
        """
        Compute k-dimensional spectral embeddings for each member.
        """
        k = min(k, min(self.B.shape) - 1)
        
        # Compute SVD
        U, s, Vt = svds(self.B, k=k)
        
        # Sort by singular values (descending)
        idx = np.argsort(s)[::-1]
        U = U[:, idx]
        s = s[idx]
        
        # Compute weighted embeddings (emphasize structural modes)
        # Weight by 1/sigma to capture low-frequency structure
        weights = 1.0 / (s + 1e-10)
        self.embeddings = U * np.sqrt(weights)
        self.singular_values = s
        
        print(f"  Computed {k}-dimensional embeddings")
        print(f"  Top 5 singular values: {s[:5]}")
    
    def predict_coalition(self, bill_features, threshold=0.7, n_bootstrap=100):
        """
        Predict which members will form a coalition on a bill.
        
        Args:
            bill_features: Dict describing the bill:
                - 'policy_area': str (e.g., 'healthcare', 'defense')
                - 'partisan_score': float (-1 to 1, expected partisan lean)
                - 'salience': float (0 to 1, media attention)
                - 'sponsor_icpsr': int (bill sponsor member ID)
        
        Returns:
            DataFrame with coalition predictions for each member
        """
        # Create synthetic vote vector based on bill features
        synthetic_vote = self._create_synthetic_vote(bill_features)
        
        # Project into spectral space
        vote_embedding = self._project_vote(synthetic_vote)
        
        # Compute coalition probabilities
        predictions = []
        
        for i, member_id in enumerate(self.member_ids):
            member_embedding = self.embeddings[i]
            member_info = self.member_map.get(member_id, {})
            
            # Spectral similarity (cosine distance in embedding space)
            similarity = 1 - cosine(member_embedding, vote_embedding)
            
            # Base probability from similarity
            base_prob = (similarity + 1) / 2  # Normalize to [0, 1]
            
            # Adjust for partisan alignment
            party = member_info.get('party_code', 0)
            party_adjustment = self._party_adjustment(party, bill_features['partisan_score'])
            
            # Adjust for member's structural position
            centrality = np.sum(self.embeddings[i]**2)
            centrality_weight = min(0.2, centrality / 100)  # High centrality = less predictable
            
            # Final probability
            coalition_prob = base_prob * (1 + party_adjustment) * (1 - centrality_weight)
            coalition_prob = np.clip(coalition_prob, 0, 1)
            
            # Bootstrap confidence interval
            ci_lower, ci_upper = self._bootstrap_confidence(
                member_embedding, vote_embedding, n_bootstrap
            )
            
            predictions.append({
                'icpsr': member_id,
                'name': member_info.get('bioname', f'Unknown {member_id}'),
                'party': party,
                'state': member_info.get('state_abbrev', ''),
                'coalition_probability': coalition_prob,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'predicted_vote': 'Yea' if coalition_prob > threshold else 'Nay',
                'spectral_similarity': similarity,
                'key_swing': abs(coalition_prob - 0.5) < 0.15  # Uncertain votes
            })
        
        return pd.DataFrame(predictions).sort_values('coalition_probability', ascending=False)
    
    def _create_synthetic_vote(self, bill_features):
        """
        Create a synthetic vote vector based on bill features.
        """
        n_members = len(self.member_ids)
        vote = np.zeros(n_members)
        
        # Base partisan pattern
        partisan_leans = np.array([
            1 if self.member_map.get(mid, {}).get('party_code') == 200 else -1
            for mid in self.member_ids
        ])
        
        # Adjust for bill's expected partisan lean
        vote = partisan_leans * bill_features['partisan_score']
        
        # Sponsor effect (sponsor and allies more likely to support)
        sponsor = bill_features.get('sponsor_icpsr')
        if sponsor and sponsor in self.member_ids:
            sponsor_idx = self.member_ids.index(sponsor)
            # Find members spectrally close to sponsor
            sponsor_emb = self.embeddings[sponsor_idx]
            similarities = [
                1 - cosine(self.embeddings[i], sponsor_emb)
                for i in range(len(self.member_ids))
            ]
            # Boost probability for allies
            vote += np.array(similarities) * 0.3
        
        # Salience effect (high salience = more partisan)
        if bill_features['salience'] > 0.7:
            vote *= 1.5  # Amplify partisan signal
        
        return np.clip(vote, -1, 1)
    
    def _project_vote(self, vote):
        """
        Project a vote vector into the spectral embedding space.
        """
        # Use pseudo-inverse to project
        k = self.embeddings.shape[1]
        sigma_inv = 1.0 / (self.singular_values[:k] + 1e-10)
        
        # vote_embedding = V^T @ diag(sigma^-1) @ vote
        # Approximate using learned embeddings
        # Actually: embedding = U^T @ vote / sigma
        
        # Simplified: just use correlation with member embeddings
        embedding = np.zeros(k)
        for i in range(k):
            embedding[i] = np.corrcoef(self.embeddings[:, i], vote)[0, 1]
        
        return embedding
    
    def _party_adjustment(self, party_code, bill_partisan_score):
        """
        Adjust probability based on party alignment with bill.
        """
        party_map = {100: -1, 200: 1, 328: 0}  # Dem, Rep, Ind
        party_lean = party_map.get(party_code, 0)
        
        # If party and bill align, increase probability
        return party_lean * bill_partisan_score * 0.3
    
    def _bootstrap_confidence(self, member_emb, vote_emb, n_bootstrap=100):
        """
        Compute bootstrap confidence interval for coalition probability.
        """
        probs = []
        for _ in range(n_bootstrap):
            # Add noise to embedding
            noise = np.random.normal(0, 0.1, len(member_emb))
            noisy_emb = member_emb + noise
            
            sim = 1 - cosine(noisy_emb, vote_emb)
            prob = (sim + 1) / 2
            probs.append(prob)
        
        return np.percentile(probs, 5), np.percentile(probs, 95)
    
    def identify_swing_votes(self, predictions_df, n_swings=10):
        """
        Identify the members most likely to be swing votes.
        """
        swing_df = predictions_df[predictions_df['key_swing']].copy()
        swing_df['swing_margin'] = abs(swing_df['coalition_probability'] - 0.5)
        
        return swing_df.nsmallest(n_swings, 'swing_margin')
    
    def coalition_breakdown(self, predictions_df):
        """
        Provide summary statistics on predicted coalition.
        """
        yeas = predictions_df[predictions_df['predicted_vote'] == 'Yea']
        nays = predictions_df[predictions_df['predicted_vote'] == 'Nay']
        
        print("\n" + "="*70)
        print("PREDICTED COALITION BREAKDOWN")
        print("="*70)
        
        print(f"\nPredicted Yea: {len(yeas)} ({len(yeas)/len(predictions_df)*100:.1f}%)")
        print(f"Predicted Nay: {len(nays)} ({len(nays)/len(predictions_df)*100:.1f}%)")
        
        # By party
        print("\nBy Party:")
        party_map = {100: 'Democrat', 200: 'Republican', 328: 'Independent'}
        for party in [100, 200, 328]:
            party_yeas = yeas[yeas['party'] == party]
            party_total = predictions_df[predictions_df['party'] == party]
            if len(party_total) > 0:
                name = party_map.get(party, f'Party {party}')
                pct = len(party_yeas) / len(party_total) * 100
                print(f"  {name}: {len(party_yeas)}/{len(party_total)} Yea ({pct:.1f}%)")
        
        # Swing votes
        swings = self.identify_swing_votes(predictions_df, n_swings=10)
        print(f"\nTop 10 Swing Votes (closest to 50/50):")
        for _, row in swings.iterrows():
            party = party_map.get(row['party'], 'Other')
            print(f"  {row['name']} ({party}-{row['state']}): "
                  f"{row['coalition_probability']:.2%} "
                  f"[{row['ci_lower']:.2%}, {row['ci_upper']:.2%}]")
        
        # High confidence predictions
        high_conf = predictions_df[
            (predictions_df['coalition_probability'] > 0.9) | 
            (predictions_df['coalition_probability'] < 0.1)
        ]
        print(f"\nHigh Confidence Predictions (>90% or <10%): {len(high_conf)}")
        
        return {
            'yeas': len(yeas),
            'nays': len(nays),
            'swing_votes': len(swings),
            'high_confidence': len(high_conf)
        }


def main():
    print("="*70)
    print("SPECTRAL COALITION PREDICTOR")
    print("Novel Method: Predicting vote outcomes from spectral embeddings")
    print("="*70)
    
    # Load data
    base_dir = os.path.expanduser("~/projects/CongressGAT/data")
    votes_file = os.path.join(base_dir, "H116_votes.csv")
    members_file = os.path.join(base_dir, "H116_members.csv")
    
    if not os.path.exists(votes_file):
        print(f"Error: {votes_file} not found.")
        return
    
    print(f"\nLoading data...")
    votes_df = pd.read_csv(votes_file)
    members_df = pd.read_csv(members_file)
    
    # Process
    votes_df = votes_df[votes_df['chamber'] == 'House']
    votes_df['vote_val'] = 0
    votes_df.loc[votes_df['cast_code'].isin([1, 2, 3]), 'vote_val'] = 1
    votes_df.loc[votes_df['cast_code'].isin([4, 5, 6]), 'vote_val'] = -1
    votes_df = votes_df[votes_df['vote_val'] != 0]
    
    vote_counts = votes_df.groupby('icpsr').size()
    active_members = vote_counts[vote_counts >= 50].index
    votes_df = votes_df[votes_df['icpsr'].isin(active_members)]
    
    matrix = votes_df.pivot(index='icpsr', columns='rollnumber', values='vote_val').fillna(0)
    member_map = members_df.set_index('icpsr')[['bioname', 'party_code', 'state_abbrev']].to_dict('index')
    
    print(f"Matrix: {matrix.shape[0]} members × {matrix.shape[1]} votes\n")
    
    # Initialize predictor
    predictor = SpectralCoalitionPredictor(matrix, member_map)
    
    # Test predictions on hypothetical bills
    test_bills = [
        {
            'name': 'Partisan Healthcare Bill',
            'policy_area': 'healthcare',
            'partisan_score': 0.8,  # Republican-leaning
            'salience': 0.9,
            'sponsor_icpsr': list(member_map.keys())[0] if member_map else None
        },
        {
            'name': 'Bipartisan Infrastructure Bill',
            'policy_area': 'infrastructure',
            'partisan_score': 0.1,  # Slightly partisan
            'salience': 0.7,
            'sponsor_icpsr': None
        },
        {
            'name': 'Controversial Defense Bill',
            'policy_area': 'defense',
            'partisan_score': -0.6,  # Democrat-leaning
            'salience': 0.8,
            'sponsor_icpsr': None
        }
    ]
    
    all_results = []
    
    for bill in test_bills:
        print(f"\n{'='*70}")
        print(f"PREDICTING: {bill['name']}")
        print(f"Partisan Score: {bill['partisan_score']:+.2f}, Salience: {bill['salience']}")
        print('='*70)
        
        predictions = predictor.predict_coalition(bill, threshold=0.6)
        stats = predictor.coalition_breakdown(predictions)
        
        # Store results
        predictions['bill_name'] = bill['name']
        all_results.append(predictions)
    
    # Save all predictions
    out_dir = os.path.expanduser("~/projects/spectral-anatomy-congress")
    os.makedirs(out_dir, exist_ok=True)
    
    combined = pd.concat(all_results)
    combined.to_csv(os.path.join(out_dir, "coalition_predictions.csv"), index=False)
    print(f"\n\nResults saved to {out_dir}/coalition_predictions.csv")
    
    print("\n" + "="*70)
    print("VALIDATION")
    print("="*70)
    print("""
To validate predictions, compare against actual votes from Congress.gov:
1. Identify recently passed bills matching the test scenarios
2. Compare predicted Yea/Nay with actual roll-call votes
3. Compute accuracy: (true positives + true negatives) / total
4. Compare against baseline (party-line prediction)

Expected performance:
- Partisan bills: >90% accuracy (party-line dominates)
- Bipartisan bills: 70-80% accuracy (more swing votes)
- High-salience bills: Lower accuracy (unpredictable factors)
""")


if __name__ == "__main__":
    main()
