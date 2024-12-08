"""
Quick Start Guide:
-----------------
1. Installation:
   # Core dependencies
   pip install rdkit pandas swifter scikit-learn imblearn
   
   # For molecular embeddings
   pip install gensim
   pip install git+https://github.com/samoturk/mol2vec

2. Required Data Structure:
   Parent Directory/
   ├── training_smiles.csv  # Must have 'SMILES' and 'ACTIVE' columns
   ├── test_smiles.csv      # Must have 'SMILES' column
   └── models/              # Will be created automatically
       └── model_300dim.pkl # Mol2vec model (auto-downloaded)

3. Basic Usage:
   python preprocessing.py
   
   This will:
   - Download mol2vec model if needed
   - Create 'processed_data' directory
   - Generate:
     - processed_train.parquet
     - processed_test.parquet
     - selected_columns.txt

4. Output Structure:
   processed_data/
   ├── processed_train.parquet  # Processed training data
   ├── processed_test.parquet   # Processed test data
   └── selected_columns.txt     # Selected feature names

Note: First run may take longer due to model download and feature computation
"""

import os
import warnings
warnings.filterwarnings('ignore')
os.environ['MKL_DISABLE_FAST_MM'] = '1'

import logging
from functools import lru_cache
from rdkit import RDLogger
import time
from sklearn.metrics import roc_auc_score

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

from typing import List, Literal, Tuple, Optional, Dict  
from pathlib import Path


import numpy as np
import pandas as pd
import swifter
from rdkit import Chem
from rdkit.Chem import Fragments, Lipinski, rdFingerprintGenerator, Descriptors
from rdkit.Chem.Descriptors import MolLogP, rdMolDescriptors
from rdkit.Chem.rdchem import Mol
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    SelectFdr,
    SelectFpr,
    SelectFromModel,
    SelectFwe,
    SelectKBest,
    VarianceThreshold,
    mutual_info_classif,
)
from sklearn.preprocessing import MinMaxScaler
# pip install git+https://github.com/samoturk/mol2vec
from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec
from gensim.models import word2vec
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from scipy import stats
from statsmodels.stats.multitest import multipletests


def fr_fluoro(mol: Mol) -> int:
    """Count carbon-fluorine bonds in molecule"""
    return sum(
        1
        for bond in mol.GetBonds()
        if bond.GetBeginAtom().GetAtomicNum() == 6  # Carbon
        and bond.GetEndAtom().GetAtomicNum() == 9  # Fluorine
    )


def fr_chloro(mol: Mol) -> int:
    """Count carbon-chlorine bonds in molecule"""
    return sum(
        1
        for bond in mol.GetBonds()
        if bond.GetBeginAtom().GetAtomicNum() == 6  # Carbon
        and bond.GetEndAtom().GetAtomicNum() == 17  # Chlorine
    )


def fr_arom_oxo(mol: Mol) -> int:
    """Count aromatic carbon-oxygen double bonds"""
    return sum(
        1
        for bond in mol.GetBonds()
        if bond.GetBondTypeAsDouble() == 2  # Double bond
        and bond.GetBeginAtom().GetAtomicNum() == 6  # Carbon
        and bond.GetEndAtom().GetAtomicNum() == 8  # Oxygen
        and bond.GetBeginAtom().GetIsAromatic()  # Aromatic carbon
    )


def fr_alcohol(mol: Mol) -> int:
    """Count non-aromatic hydroxyl groups"""
    return sum(
        1
        for bond in mol.GetBonds()
        if bond.GetBondTypeAsDouble() == 1  # Single bond
        and bond.GetBeginAtom().GetAtomicNum() == 8  # Oxygen
        and bond.GetEndAtom().GetAtomicNum() == 1  # Hydrogen
        and bond.GetEndAtom().GetIsAromatic() == False  # Non-aromatic oxygen
    )


def fr_alkene(mol: Mol) -> int:
    """Count carbon-carbon double bonds"""
    return sum(
        1
        for bond in mol.GetBonds()
        if bond.GetBondTypeAsDouble() == 2  # Double bond
        and bond.GetBeginAtom().GetAtomicNum() == 6  # Carbon
        and bond.GetEndAtom().GetAtomicNum() == 6  # Carbon
    )


def expand_structural_features(
    df: pd.DataFrame, radius: int = 3, vector_size: int = 1024
) -> pd.DataFrame:
    """
    Comprehensive molecular feature extraction pipeline that generates:
    - Functional group counts (amides, ethers, etc.)
    - Extended Connectivity Fingerprints (ECFP) for structural similarity
    - Basic molecular descriptors (atom/bond counts)
    
    The ECFP radius parameter controls the size of structural fragments considered
    vector_size determines fingerprint length (higher = more detailed but slower)
    """
    new_df = df.copy()
    mols = df["MOL"]

    # Functional groups
    new_df["n_amides"] = mols.swifter.apply(Fragments.fr_amide)
    new_df["n_ethers"] = mols.swifter.apply(Fragments.fr_ether)
    new_df["n_teramines"] = mols.swifter.apply(Fragments.fr_NH0)
    new_df["n_secamines"] = mols.swifter.apply(Fragments.fr_NH1)
    new_df["n_fluoros"] = mols.swifter.apply(fr_fluoro)
    new_df["n_chloros"] = mols.swifter.apply(fr_chloro)
    new_df["n_carboxacids"] = mols.swifter.apply(Fragments.fr_COO)
    new_df["n_oxos"] = mols.swifter.apply(fr_arom_oxo)
    new_df["n_alcohols"] = mols.swifter.apply(fr_alcohol)
    new_df["n_phenols"] = mols.swifter.apply(Fragments.fr_phenol)
    new_df["n_sulfoanamids"] = mols.swifter.apply(Fragments.fr_sulfonamd)
    new_df["n_priamines"] = mols.swifter.apply(Fragments.fr_NH2)
    new_df["n_nitriles"] = mols.swifter.apply(Fragments.fr_nitrile)
    new_df["n_alkenes"] = mols.swifter.apply(fr_alkene)
    new_df["n_anilines"] = mols.swifter.apply(Fragments.fr_aniline)
    new_df["n_halogens"] = mols.swifter.apply(Fragments.fr_halogen)
    new_df["n_alihydroxyls"] = mols.swifter.apply(Fragments.fr_Al_OH)
    new_df["n_arohydroxyls"] = mols.swifter.apply(Fragments.fr_Ar_OH)

    # Additional functional groups
    new_df["n_benzene"] = mols.swifter.apply(Fragments.fr_benzene)
    new_df["n_ester"] = mols.swifter.apply(Fragments.fr_ester)
    new_df["n_nitro"] = mols.swifter.apply(Fragments.fr_nitro)
    new_df["n_nitro_arom"] = mols.swifter.apply(Fragments.fr_nitro_arom)
    new_df["n_al_coo"] = mols.swifter.apply(Descriptors.fr_Al_COO)
    new_df["n_ar_n"] = mols.swifter.apply(Fragments.fr_Ar_N)

    # Additional atom counts
    new_df["n_nhoh"] = mols.swifter.apply(Lipinski.NHOHCount)
    new_df["n_no"] = mols.swifter.apply(Lipinski.NOCount)

    # ECFP
    ecfpgen = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=vector_size
    )
    new_df["ecfp"] = mols.swifter.apply(
        lambda x: ecfpgen.GetFingerprintAsNumPy(x).tolist()
    )
    new_df = pd.concat(
        [
            new_df,
            pd.DataFrame(
                new_df["ecfp"].to_list(),
                index=new_df.index,
                columns=[f"ecfp_{i}" for i in range(vector_size)],
            ),
        ],
        axis=1,
    )
    new_df = new_df.drop(columns=["ecfp"])

    # Basic molecular information
    new_df["n_atoms"] = mols.swifter.apply(rdMolDescriptors.CalcNumAtoms)
    new_df["n_heavyatoms"] = mols.swifter.apply(rdMolDescriptors.CalcNumHeavyAtoms)
    new_df["n_bonds"] = mols.swifter.apply(lambda x: x.GetNumBonds())

    return new_df


def expand_physicochemical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates key physicochemical properties:
    - Molecular weight
    - Lipophilicity (LogP)
    - H-bond donors/acceptors
    - Rotatable bonds
    """
    new_df = df.copy()
    mols = new_df["MOL"]

    # Weight
    new_df["exact_weight"] = mols.swifter.apply(rdMolDescriptors.CalcExactMolWt)
    # Lipophilicity
    new_df["log_p"] = mols.swifter.apply(MolLogP)
    # Hydrogen bonding capacity
    new_df["n_donors"] = mols.swifter.apply(Lipinski.NumHDonors)
    new_df["n_acceptors"] = mols.swifter.apply(Lipinski.NumHAcceptors)
    # Rotatable bonds
    new_df["n_rotbonds"] = mols.swifter.apply(rdMolDescriptors.CalcNumRotatableBonds)

    return new_df


def expand_electronic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes electronic/structural properties:
    - Topological polar surface area (TPSA)
    - Number of aromatic rings
    """
    new_df = df.copy()
    mols = new_df["MOL"]

    # Polarity and charge
    new_df["TPSA"] = mols.swifter.apply(
        rdMolDescriptors.CalcTPSA
    )  # Topological polar surface area
    # Aromatic rings
    new_df["n_arorings"] = mols.swifter.apply(rdMolDescriptors.CalcNumAromaticRings)
    # Eletronic density by number of valence electrons
    new_df["e_density"] = mols.swifter.apply(Descriptors.NumValenceElectrons)

    # Additional ring counts
    new_df["n_aliphatic_rings"] = mols.swifter.apply(rdMolDescriptors.CalcNumAliphaticRings)
    new_df["n_saturated_rings"] = mols.swifter.apply(rdMolDescriptors.CalcNumSaturatedRings)
    
    # Additional electronic properties
    new_df["n_radical_electrons"] = mols.swifter.apply(Descriptors.NumRadicalElectrons)

    return new_df


def expand_mol2vec_features(df: pd.DataFrame, model_path: str = "models/model_300dim.pkl") -> pd.DataFrame:
    """
    Generates molecular embeddings using Mol2Vec, which converts molecules into vector representations:
    - Uses pre-trained word2vec model on molecular substructures
    - Each molecule gets a 300-dimensional embedding vector
    - Handles missing substructures by averaging available vectors
    - Useful for capturing complex molecular similarities
    """
    t_start = time.time()
    
    # Load pre-trained mol2vec model
    logging.info("Loading mol2vec model...")
    model = word2vec.Word2Vec.load(model_path)
    
    # Construct molecular sentences
    logging.info(f"Processing {len(df)} molecules for mol2vec...")
    sentences = [MolSentence(mol2alt_sentence(mol, 1)) for mol in df['MOL']]
    
    # Generate embeddings (adapted for gensim 4.x)
    logging.info("Generating embeddings...")
    keys = set(model.wv.index_to_key)
    vectors = []
    for i, sentence in enumerate(sentences):
        if i % 10000 == 0 and i > 0:
            logging.info(f"Processed {i}/{len(sentences)} molecules...")
        sent_vec = np.zeros(model.vector_size)
        count = 0
        for word in sentence:
            if word in keys:
                sent_vec += model.wv[word]
                count += 1
        if count > 0:
            sent_vec /= count
        vectors.append(sent_vec)
    
    # Create feature matrix
    logging.info("Creating feature matrix...")
    mol2vec_columns = [f"mol2vec_{i}" for i in range(model.vector_size)]
    mol2vec_df = pd.DataFrame(vectors, columns=mol2vec_columns, index=df.index)
    
    elapsed = time.time() - t_start
    logging.info(f"Mol2vec processing completed in {elapsed:.2f} seconds")
    
    return pd.concat([df, mol2vec_df], axis=1)


def expand_dataset(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Starting feature expansion...")
    new_df = df.copy()
    
    # Convert SMILES to MOL objects
    logging.info("Converting SMILES to MOL objects...")
    try:
        new_df["MOL"] = new_df["SMILES"].swifter.apply(Chem.MolFromSmiles)
    except ImportError:
        new_df["MOL"] = new_df["SMILES"].apply(Chem.MolFromSmiles)
    
    new_df = new_df.drop(columns=["SMILES"])
    
    logging.info("Generating structural features...")
    new_df = expand_structural_features(new_df)
    
    logging.info("Generating physicochemical features...")
    new_df = expand_physicochemical_features(new_df)
    
    logging.info("Generating electronic features...")
    new_df = expand_electronic_features(new_df)
    
    logging.info("Generating mol2vec features...")
    new_df = expand_mol2vec_features(new_df)
    
    new_df = new_df.drop(columns=["MOL"])
    logging.info("Feature expansion complete.")

    return new_df


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=["ACTIVE"]), df["ACTIVE"]


class BasePreprocessing:
    """
    Core preprocessing pipeline implementing:
    1. Feature expansion from SMILES strings to molecular descriptors
    2. Removal of constant/non-informative features
    3. Feature scaling to [0,1] range
    4. Dimensionality reduction of ECFP fingerprints via PCA
    
    This serves as foundation for more advanced preprocessing strategies
    """
    def __init__(self):
        self.constant_remover: VarianceThreshold = None
        self.feature_scaler: MinMaxScaler = None
        self.ecfp_cols: List[str] = None
        self.pca: PCA = None
        self.pca_scaler: MinMaxScaler = None

    def fit_transform(self, df: pd.DataFrame, expand: bool = False) -> pd.DataFrame:
        """Base preprocessing pipeline"""
        t_start = time.time()
        logging.info("Starting fit_transform...")
        
        if expand:
            logging.info("Expanding dataset features...")
            new_df = expand_dataset(df)
            logging.info(f"Dataset expanded: {new_df.shape}")
        else:
            new_df = df.copy()
        
        logging.info("Fitting constant remover...")
        new_df = self.__fit_constant_remover(new_df)
        logging.info(f"After constant removal: {new_df.shape}")
        
        logging.info("Removing duplicates...")
        new_df = new_df.drop_duplicates(keep="first")
        logging.info(f"After duplicate removal: {new_df.shape}")
        
        logging.info("Fitting data scaler...")
        new_df = self.__fit_data_scaler(new_df)
        logging.info(f"After scaling: {new_df.shape}")
        
        logging.info("Fitting ECFP PCA...")
        new_df = self.__fit_ecfp_pca(new_df)
        logging.info(f"After ECFP PCA: {new_df.shape}")
        
        elapsed = time.time() - t_start
        logging.info(f"fit_transform completed in {elapsed:.2f} seconds")
        return new_df

    def __fit_ecfp_pca(self, df: pd.DataFrame) -> Tuple[List[str], PCA, pd.DataFrame]:
        """
        Reduces high-dimensional ECFP fingerprints:
        1. Identifies ECFP columns in dataset
        2. Applies PCA to reduce to 100 most important components
        3. Scales PCA components to [0,1] for consistency
        
        Returns modified dataframe with original ECFP columns replaced by PCA components
        """
        new_df = df.copy()

        self.ecfp_cols = [col for col in new_df.columns if col.startswith("ecfp")]
        ecfp_set = new_df[self.ecfp_cols]

        self.pca = PCA(100)
        Z_ecfp = self.pca.fit_transform(ecfp_set)
        Z_ecfp = pd.DataFrame(
            Z_ecfp, index=new_df.index, columns=[f"z_ecfp_{i}" for i in range(100)]
        )

        self.pca_scaler = MinMaxScaler()
        Z_ecfp = pd.DataFrame(
            self.pca_scaler.fit_transform(Z_ecfp),
            columns=Z_ecfp.columns,
            index=Z_ecfp.index,
        )

        return pd.concat([new_df, Z_ecfp], axis=1).drop(columns=self.ecfp_cols)

    def __fit_data_scaler(self, df: pd.DataFrame) -> pd.DataFrame:
        new_df = df.copy()

        self.feature_scaler = MinMaxScaler()
        X, y = split_features_target(new_df)
        X = pd.DataFrame(
            self.feature_scaler.fit_transform(X.values),
            columns=X.columns,
            index=X.index,
        )

        return pd.concat([X, y], axis=1)

    def __fit_constant_remover(self, df: pd.DataFrame) -> pd.DataFrame:
        new_df = df.copy()

        X, y = split_features_target(new_df)
        self.constant_remover = VarianceThreshold().fit(X, y)
        new_df: pd.DataFrame = new_df[
            list(self.constant_remover.get_feature_names_out()) + ["ACTIVE"]
        ]

        return new_df

    def transform(self, feature_df: pd.DataFrame, expand: bool = False) -> pd.DataFrame:
        if expand:
            new_df = expand_dataset(feature_df)
        else:
            new_df = feature_df.copy()

        new_df = self.__apply_constant_remover(new_df)

        new_df = self.__apply_data_scaler(new_df)

        new_df = self.__apply_ecfp_pca(new_df)

        return new_df

    def __apply_ecfp_pca(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        new_df = feature_df.copy()

        ecfp_set = new_df[self.ecfp_cols]

        Z_ecfp = self.pca.transform(ecfp_set)
        Z_ecfp = pd.DataFrame(
            Z_ecfp, index=new_df.index, columns=[f"z_ecfp_{i}" for i in range(100)]
        )

        Z_ecfp = pd.DataFrame(
            self.pca_scaler.transform(Z_ecfp),
            columns=Z_ecfp.columns,
            index=Z_ecfp.index,
        )

        return pd.concat([new_df, Z_ecfp], axis=1).drop(columns=self.ecfp_cols)

    def __apply_data_scaler(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        X = feature_df.copy()
        X = pd.DataFrame(
            self.feature_scaler.transform(X.values),
            columns=X.columns,
            index=X.index,
        )
        return X

    def __apply_constant_remover(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        new_df = feature_df.copy()

        try:
            # Get feature names directly from support mask if get_feature_names_out fails
            try:
                selected_features = list(self.constant_remover.get_feature_names_out())
            except (AttributeError, TypeError):
                feature_mask = self.constant_remover.get_support()
                selected_features = feature_df.columns[feature_mask].tolist()
            
            new_df = new_df[selected_features]
            
        except Exception as e:
            logging.error(f"Error in constant remover: {e}")
            # Fallback: return all features if constant remover fails
            new_df = feature_df.copy()
        
        return new_df


class StatisticalPreprocessing:
    """
    Advanced feature selection pipeline using multiple statistical approaches:
    - K-best features based on mutual information
    - False Positive Rate (FPR) control
    - False Discovery Rate (FDR) control
    - Family-wise Error Rate (FWER) control
    - Model-based selection using Random Forest importance
    
    Automatically evaluates and selects best performing method via cross-validation
    """
    def __init__(self, transformed_df: Optional[pd.DataFrame] = None) -> None:
        self.columns: Dict[str, List[str]] = {}
        self.base_transform: BasePreprocessing = BasePreprocessing()
        self.transformed_df = transformed_df

    def fit_transform_all_modes(
        self,
        df: pd.DataFrame,
        max_features: int = 300,
        expand: bool = False,
        sample_size: int = 50000
    ) -> Dict[str, pd.DataFrame]:
        """
        Comprehensive feature selection process:
        1. Samples data for computational efficiency
        2. Applies multiple selection methods
        3. Evaluates each method using Random Forest + cross-validation
        4. Tracks best performing method
        
        Returns dictionary of processed datasets for each selection method
        """
        logging.info(f"Starting fit_transform_all_modes with max_features={max_features}")
        
        if self.transformed_df is None:
            self.transformed_df = self.base_transform.fit_transform(df, expand)
            self.transformed_df.to_parquet("transformed_df.parquet")

        results = {}
        X, y = split_features_target(self.transformed_df)
        
        # Sample data for faster processing
        logging.info(f"Sampling {sample_size} rows for feature selection...")
        sample_idx = np.random.choice(len(X), size=min(sample_size, len(X)), replace=False)
        X_sample = X.iloc[sample_idx]
        y_sample = y.iloc[sample_idx]
        
        for mode in ["nbest", "FPR", "FDR", "FWER", "model_based"]:
            try:
                logging.info(f"\nProcessing mode: {mode}")
                
                # Calculate MI scores once
                mi_scores = mutual_info_classif(X_sample, y_sample)
                scores = pd.Series(mi_scores, index=X_sample.columns)
                
                # Convert MI scores to p-values
                p_values = 1 - stats.norm.cdf(stats.zscore(mi_scores))
                
                # Select features based on mode
                if mode == "nbest":
                    selected_features = scores.nlargest(max_features).index.tolist()
                
                elif mode == "model_based":
                    selector = SelectFromModel(
                        RandomForestClassifier(n_estimators=100, random_state=42),
                        max_features=max_features
                    )
                    selector.fit(X_sample, y_sample)
                    selected_features = X_sample.columns[selector.get_support()].tolist()
                
                else:
                    # Apply different statistical tests
                    if mode == "FPR":
                        # Control false positive rate
                        threshold = p_values < 0.6
                    elif mode == "FDR":
                        reject, _, _, _ = multipletests(p_values, method='fdr_bh', alpha=0.8)
                        threshold = reject
                    else:  # FWER
                        reject, _, _, _ = multipletests(p_values, method='bonferroni', alpha=0.6)
                        threshold = reject
                    
                    selected_features = X_sample.columns[threshold].tolist()
                    
                    # Fallback if too few features
                    if len(selected_features) < 10:
                        logging.info(f"Too few features selected ({len(selected_features)}), falling back to top {max_features}")
                        selected_features = scores.nlargest(max_features).index.tolist()
                
                logging.info(f"Mode: {mode}")
                logging.info(f"Features selected: {len(selected_features)}")
                
                # Evaluate feature set
                X_selected = X_sample[selected_features]
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                scores = cross_val_score(clf, X_selected, y_sample, cv=cv, scoring='roc_auc')
                
                mean_auc = np.mean(scores)
                std_auc = np.std(scores)
                
                logging.info(f"Mean AUC-ROC: {mean_auc:.3f} ± {std_auc:.3f}")
                
                # Store results
                self.columns[mode] = selected_features
                results[mode] = self.transformed_df[selected_features + ['ACTIVE']]
                
                # Track best performance
                if mean_auc > getattr(self, 'best_auc', 0):
                    self.best_auc = mean_auc
                    self.best_mode = mode
                    
            except Exception as e:
                logging.error(f"Error in mode {mode}: {str(e)}")
                continue
        
        return results

    def transform(
        self, 
        feature_df: pd.DataFrame, 
        mode: Literal["nbest", "FPR", "FDR", "FWER"],
        expand: bool = False
    ) -> pd.DataFrame:
        new_df = self.base_transform.transform(feature_df, expand)
        return new_df[self.columns[mode]]


def compute_score(selector: BaseEstimator):
    """
    Extracts feature importance scores from a fitted model
    Handles both linear models (coef_) and tree-based models (feature_importances_)
    """
    try:
        importance = np.absolute(selector.estimator_.coef_).squeeze()
    except AttributeError:
        importance = selector.estimator_.feature_importances_

    return importance


def download_mol2vec_model(model_path: str = "models/model_300dim.pkl") -> None:
    """
    Downloads pretrained Mol2vec model if not present
    """
    import urllib.request
    import os
    
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    if not os.path.exists(model_path):
        url = "https://raw.githubusercontent.com/samoturk/mol2vec/master/examples/models/model_300dim.pkl"
        urllib.request.urlretrieve(url, model_path)


def main():
    """
    End-to-end preprocessing pipeline:
    1. Downloads required Mol2Vec model
    2. Loads training/test data
    3. Applies feature engineering and selection
    4. Evaluates feature selection methods
    5. Saves processed datasets and selected features
    
    Handles caching of intermediate results for efficiency
    """
    try:
        # Download Mol2vec model if needed
        model_path = "models/model_300dim.pkl"
        download_mol2vec_model(model_path)
        
        # Get data paths
        current_dir = Path(__file__).parent
        assignment_dir = current_dir.parent
        output_dir = assignment_dir / "processed_data"
        output_dir.mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Load data
        train_df = pd.read_csv(assignment_dir / "training_smiles.csv")
        test_df = pd.read_csv(assignment_dir / "test_smiles.csv")
        
        # Process features and evaluate
        logging.info("Running feature selection and evaluation...")
        transformed_df = None
        if os.path.exists("transformed_df.parquet"):
            transformed_df = pd.read_parquet("transformed_df.parquet")
            logging.info("Loaded pre-computed transformed data")
        
        # Process training data - first fit the base transform
        stat_preprocessor = StatisticalPreprocessing(transformed_df=transformed_df)
        if transformed_df is None:
            # This will fit the base_transform
            feature_sets = stat_preprocessor.fit_transform_all_modes(
                train_df,
                max_features=300,
                expand=True
            )
        else:
            # Need to ensure base_transform is fitted even when using cached transformed_df
            stat_preprocessor.base_transform.fit_transform(train_df, expand=True)
            feature_sets = stat_preprocessor.fit_transform_all_modes(
                train_df,
                max_features=300,
                expand=False  # Already expanded in base_transform
            )
        
        # Get best columns and save transformed training data
        best_columns = stat_preprocessor.columns[stat_preprocessor.best_mode]
        transformed_train = stat_preprocessor.transformed_df[best_columns + ['ACTIVE']]
        transformed_train.to_parquet(output_dir / "processed_train.parquet")
        
        # Now transform test data using fitted base_transform
        transformed_test = stat_preprocessor.base_transform.transform(
            test_df, 
            expand=True
        )[best_columns]
        
        # Save processed test data
        transformed_test.to_parquet(output_dir / "processed_test.parquet")
        
        # Save only the best performing feature set
        best_mode = stat_preprocessor.best_mode
        best_auc = stat_preprocessor.best_auc
        best_columns = stat_preprocessor.columns[best_mode]
        
        logging.info(f"\nBest performing mode: {best_mode}")
        logging.info(f"AUC-ROC: {best_auc:.3f}")
        logging.info(f"Number of selected features: {len(best_columns)}")
        
        # Save selected columns to file
        output_dir = Path(__file__).parent.parent / "processed_data"
        output_dir.mkdir(exist_ok=True)
        
        columns_file = output_dir / "selected_columns.txt"
        with open(columns_file, 'w') as f:
            f.write('\n'.join(best_columns))
            
        logging.info(f"\nSaved selected columns to: {columns_file}")
        
    except Exception as e:
        logging.error(f"Fatal error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()