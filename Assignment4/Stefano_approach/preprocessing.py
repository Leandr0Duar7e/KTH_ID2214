import os
import warnings
import urllib.request
import logging
from functools import lru_cache

# Suppress Intel MKL warnings
os.environ['MKL_DISABLE_FAST_MM'] = '1'

# Suppress all warnings
warnings.filterwarnings('ignore')

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
from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec
from gensim.models import word2vec


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
    Extracts structural features from molecules including:
    - Functional group counts (amides, ethers, amines etc.)
    - Extended Connectivity Fingerprints (ECFP)
    - Basic molecular information (atom counts, bonds)
    
    Parameters:
        radius: ECFP radius parameter
        vector_size: Length of ECFP bit vector
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
    Computes Mol2vec embeddings for molecules
    
    Parameters:
        df: DataFrame with MOL column
        model_path: Path to pretrained Mol2vec model
    
    Returns:
        DataFrame with added Mol2vec features
    """
    new_df = df.copy()
    mols = new_df["MOL"]
    
    # Load pretrained model
    model = word2vec.Word2Vec.load(model_path)
    
    # Convert molecules to sentences and then to vectors
    vectors = []
    for mol in mols:
        # Get sentence
        sentence = mol2alt_sentence(mol, 1)
        # Convert sentence to numpy array before creating DfVec
        sentence = np.array(sentence)
        # Get vector
        try:
            vec = DfVec(sentence).vec(model)
            vectors.append(vec)
        except:
            # If there's an error, append zeros
            vectors.append(np.zeros(300))
    
    # Convert to numpy array
    vectors = np.array(vectors)
    
    # Create feature columns
    mol2vec_features = pd.DataFrame(
        vectors,
        columns=[f"mol2vec_{i}" for i in range(300)],
        index=new_df.index
    )
    
    return pd.concat([new_df, mol2vec_features], axis=1)


def expand_dataset(df: pd.DataFrame) -> pd.DataFrame:
    new_df = df.copy()
    
    # Try using swifter if available, fallback to regular apply if not
    try:
        new_df["MOL"] = new_df["SMILES"].swifter.apply(Chem.MolFromSmiles)
    except ImportError:
        new_df["MOL"] = new_df["SMILES"].apply(Chem.MolFromSmiles)
    
    new_df = new_df.drop(columns=["SMILES"])
    new_df = expand_structural_features(new_df)
    new_df = expand_physicochemical_features(new_df)
    new_df = expand_electronic_features(new_df)
    new_df = expand_mol2vec_features(new_df)
    new_df = new_df.drop(columns=["MOL"])

    return new_df


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=["ACTIVE"]), df["ACTIVE"]


class BasePreprocessing:
    """
    Base preprocessing pipeline that handles:
    1. Feature expansion from SMILES
    2. Removal of constant features
    3. Feature scaling
    4. Dimensionality reduction of ECFP via PCA
    """
    def __init__(self):
        self.constant_remover: VarianceThreshold = None
        self.feature_scaler: MinMaxScaler = None
        self.ecfp_cols: List[str] = None
        self.pca: PCA = None
        self.pca_scaler: MinMaxScaler = None

    def fit_transform(self, df: pd.DataFrame, expand: bool = False) -> pd.DataFrame:
        if expand:
            new_df = expand_dataset(df)
        else:
            new_df = df.copy()

        new_df = self.__fit_constant_remover(new_df)

        new_df = new_df.drop_duplicates(keep="first")

        new_df = self.__fit_data_scaler(new_df)

        new_df = self.__fit_ecfp_pca(new_df)

        return new_df

    def __fit_ecfp_pca(self, df: pd.DataFrame) -> Tuple[List[str], PCA, pd.DataFrame]:
        """
        Reduces ECFP dimensionality using PCA:
        1. Extracts ECFP columns
        2. Applies PCA to reduce to 100 components
        3. Scales PCA components to [0,1] range
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

        new_df: pd.DataFrame = new_df[
            list(self.constant_remover.get_feature_names_out())
        ]

        return new_df


class StatisticalPreprocessing:
    """
    Extends BasePreprocessing with statistical feature selection using:
    - K-best features
    - False Positive Rate control
    - False Discovery Rate control 
    - Family-wise Error Rate control
    
    All based on mutual information with target variable
    """
    def __init__(self) -> None:
        self.columns: Dict[str, List[str]] = {}  # Store columns for each mode
        self.base_transform: BasePreprocessing = BasePreprocessing()
        self.transformed_df: Optional[pd.DataFrame] = None  # Cache transformed data

    def fit_transform_all_modes(
        self,
        df: pd.DataFrame,
        max_features: int = 10,
        expand: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fit and transform data for all modes at once, reusing feature generation
        """
        # Generate features only once
        if self.transformed_df is None:
            self.transformed_df = self.base_transform.fit_transform(df, expand)

        X, y = split_features_target(self.transformed_df)
        
        # Define all selectors
        selectors = {
            "nbest": SelectKBest(score_func=mutual_info_classif, k=max_features),
            "FPR": SelectFpr(score_func=mutual_info_classif, alpha=1),
            "FDR": SelectFdr(score_func=mutual_info_classif, alpha=1),
            "FWER": SelectFwe(score_func=mutual_info_classif, alpha=1)
        }
        
        # Fit all modes at once
        results = {}
        for mode, selector in selectors.items():
            selector.fit(X, y)
            scores = pd.Series(selector.scores_, index=selector.feature_names_in_)
            self.columns[mode] = list(scores.nlargest(max_features, keep="all").index)
            results[mode] = self.transformed_df[self.columns[mode]]
            
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


class FromModelPreprocessing:
    """
    Extends BasePreprocessing with model-based feature selection:
    - Uses any sklearn-compatible model's feature importance
    - Selects top N most important features
    - Can reuse molecular features from StatisticalPreprocessing
    """
    def __init__(self) -> None:
        self.base_transform: BasePreprocessing = BasePreprocessing()
        self.columns: List[str] = None
        self.transformed_df: Optional[pd.DataFrame] = None

    def fit_transform(
        self,
        df: pd.DataFrame,
        model: BaseEstimator,
        max_features: int = 10,
        expand: bool = False,
        precomputed_features: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Parameters:
            precomputed_features: Optional preprocessed features from StatisticalPreprocessing
                                to avoid regenerating molecular features
        """
        # Reuse precomputed features if available
        if precomputed_features is not None:
            self.transformed_df = precomputed_features
        else:
            self.transformed_df = self.base_transform.fit_transform(df, expand)

        X, y = split_features_target(self.transformed_df)

        selector = SelectFromModel(model)
        selector = selector.fit(X, y)
        importance = compute_score(selector)

        scores = pd.Series(data=importance, index=selector.feature_names_in_)
        self.columns = list(scores.nlargest(max_features, keep="all").index)

        return self.transformed_df[self.columns]

    def transform(self, feature_df: pd.DataFrame, expand: bool = False) -> pd.DataFrame:
        new_df = self.base_transform.transform(feature_df, expand)
        return new_df[self.columns]


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
    Process and save datasets for modeling
    """
    # Download Mol2vec model if needed
    model_path = "models/model_300dim.pkl"
    download_mol2vec_model(model_path)
    
    # Get data paths
    current_dir = Path(__file__).parent
    assignment_dir = current_dir.parent
    
    # Load raw data
    train_df = pd.read_csv(assignment_dir / "training_smiles.csv")
    test_df = pd.read_csv(assignment_dir / "test_smiles.csv")
    
    # Create output directory
    output_dir = assignment_dir / "processed_data"
    output_dir.mkdir(exist_ok=True)
    
    # 1. Statistical Feature Selection
    print("\nRunning statistical feature selection...")
    stat_preprocessor = StatisticalPreprocessing()
    feature_sets = stat_preprocessor.fit_transform_all_modes(
        train_df,
        max_features=100,
        expand=True
    )
    
    # Save statistical feature sets
    for mode, train_features in feature_sets.items():
        # Save training data
        train_features.to_parquet(output_dir / f"train_stat_{mode}.parquet")
        
        # Process and save test data
        test_features = stat_preprocessor.transform(test_df, mode, expand=True)
        test_features.to_parquet(output_dir / f"test_stat_{mode}.parquet")
        
        print(f"\nStatistical Mode: {mode}")
        print(f"Training shape: {train_features.shape}")
        print(f"Test shape: {test_features.shape}")
    
    # 2. Model-based Feature Selection
    print("\nRunning model-based feature selection...")
    from sklearn.ensemble import RandomForestClassifier
    
    model_preprocessor = FromModelPreprocessing()
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Reuse features from statistical preprocessing
    train_features = model_preprocessor.fit_transform(
        train_df,
        model=rf_model,
        max_features=100,
        expand=True,
        precomputed_features=stat_preprocessor.transformed_df
    )
    
    # Transform test data
    test_features = model_preprocessor.transform(test_df, expand=True)
    
    # Save model-based features
    train_features.to_parquet(output_dir / "train_model_rf.parquet")
    test_features.to_parquet(output_dir / "test_model_rf.parquet")
    
    print("\nModel-based Selection (RF):")
    print(f"Training shape: {train_features.shape}")
    print(f"Test shape: {test_features.shape}")

if __name__ == "__main__":
    main()