import math
from Bio.SeqUtils.ProtParam import ProteinAnalysis

def bp_molecular_weight(seq):
    """
    Calculate the molecular weight of a given sequence
    """
    X = ProteinAnalysis(seq)
    return X.molecular_weight()

def bp_aromaticity(seq):
    """
    Calculate the aromaticity of a given sequence
    """
    X = ProteinAnalysis(seq)
    return X.aromaticity()

def bp_instability_index(seq):
    """
    Calculate the instability index of a given sequence
    """
    X = ProteinAnalysis(seq)
    return X.instability_index()

def bp_isoelectric_point(seq):
    """
    Calculate the isoelectric point of a given sequence
    """
    X = ProteinAnalysis(seq)
    return X.isoelectric_point()

def bp_average_flexibility(seq):
    """
    Calculate the isoelectric point of a given sequence
    """
    X = ProteinAnalysis(seq)
    return sum(X.flexibility()) / len(X.flexibility())

def bp_gravy(seq):
    """
    Calculate the isoelectric point of a given sequence
    """
    X = ProteinAnalysis(seq)
    return X.gravy()

def bp_charge_at_7_4(seq):
    """
    Calculate the isoelectric point of a given sequence
    """
    X = ProteinAnalysis(seq)
    return X.charge_at_pH(7.4)










def calculate_protein_scores(seq):
    """
    Calculates various protein scores for a given sequence.
    Returns a dictionary of calculated scores.
    """
    X = ProteinAnalysis(seq)
    
    scores = {
        "molecular_weight": X.molecular_weight(),
        "aromaticity": X.aromaticity(),
        "instability_index": X.instability_index(),
        "isoelectric_point": X.isoelectric_point(),
        "average_flexibility": sum(X.flexibility()) / len(X.flexibility()),
        "gravy": X.gravy(),
        "charge_at_7_4": X.charge_at_pH(7.4),
    }
    
    # Secondary structure fractions
    helix, turn, sheet = X.secondary_structure_fraction()
    scores["helix"] = helix
    scores["turn"] = turn
    scores["sheet"] = sheet

    # Molar extinction coefficients (reduced cysteines and oxidized disulfide bridges)
    reduced, oxidized = X.molar_extinction_coefficient()
    scores["molar_extinction_reduced"] = reduced
    scores["molar_extinction_oxidized"] = oxidized

    return scores


def biopython_score(df):
    """
    input: df with columns: heavy, light, fitness
    output: df with columns: heavy, light, fitness, 
            molecular_weight_heavy, molecular_weight_light, average_molecular_weight,
            aromaticity_heavy, aromaticity_light, average_aromaticity,
            etc.
    """
    # Initialize empty lists to store results
    scores_heavy = []
    scores_light = []

    # Calculate the scores for heavy and light chains
    for heavy_seq, light_seq in zip(df['heavy'], df['light']):
        heavy_scores = calculate_protein_scores(heavy_seq)
        light_scores = calculate_protein_scores(light_seq)

        # Append heavy and light scores
        scores_heavy.append(heavy_scores)
        scores_light.append(light_scores)

    # Create a new DataFrame to store the scores
    scores_df = pd.DataFrame(scores_heavy)
    scores_light_df = pd.DataFrame(scores_light)

    # Combine the results with the original DataFrame
    df = pd.concat([df, scores_df.add_prefix('heavy_'), scores_light_df.add_prefix('light_')], axis=1)

    # Calculate average scores for each feature
    df['average_molecular_weight'] = (df['heavy_molecular_weight'] + df['light_molecular_weight']) / 2
    df['average_aromaticity'] = (df['heavy_aromaticity'] + df['light_aromaticity']) / 2
    df['average_instability_index'] = (df['heavy_instability_index'] + df['light_instability_index']) / 2
    df['average_isoelectric_point'] = (df['heavy_isoelectric_point'] + df['light_isoelectric_point']) / 2
    df['average_flexibility'] = (df['heavy_average_flexibility'] + df['light_average_flexibility']) / 2
    df['average_gravy'] = (df['heavy_gravy'] + df['light_gravy']) / 2
    df['average_charge_at_7_4'] = (df['heavy_charge_at_7_4'] + df['light_charge_at_7_4']) / 2
    df['average_helix'] = (df['heavy_helix'] + df['light_helix']) / 2
    df['average_turn'] = (df['heavy_turn'] + df['light_turn']) / 2
    df['average_sheet'] = (df['heavy_sheet'] + df['light_sheet']) / 2
    df['average_molar_extinction_reduced'] = (df['heavy_molar_extinction_reduced'] + df['light_molar_extinction_reduced']) / 2
    df['average_molar_extinction_oxidized'] = (df['heavy_molar_extinction_oxidized'] + df['light_molar_extinction_oxidized']) / 2

    return df

