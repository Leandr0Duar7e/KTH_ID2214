from typing import List, Literal, Tuple

import numpy as np
import pandas as pd
import swifter
from rdkit import Chem
from rdkit.Chem import Fragments, Lipinski, rdFingerprintGenerator
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

swifter.__name__


def fr_fluoro(mol: Mol) -> int:
    return sum(
        1
        for bond in mol.GetBonds()
        if bond.GetBeginAtom().GetAtomicNum() == 6  # Carbon
        and bond.GetEndAtom().GetAtomicNum() == 9  # Fluorine
    )


def fr_chloro(mol: Mol) -> int:
    return sum(
        1
        for bond in mol.GetBonds()
        if bond.GetBeginAtom().GetAtomicNum() == 6  # Carbon
        and bond.GetEndAtom().GetAtomicNum() == 17  # Chlorine
    )


def fr_arom_oxo(mol: Mol) -> int:
    return sum(
        1
        for bond in mol.GetBonds()
        if bond.GetBondTypeAsDouble() == 2  # Double bond
        and bond.GetBeginAtom().GetAtomicNum() == 6  # Carbon
        and bond.GetEndAtom().GetAtomicNum() == 8  # Oxygen
        and bond.GetBeginAtom().GetIsAromatic()  # Aromatic carbon
    )


def fr_alcohol(mol: Mol) -> int:
    return sum(
        1
        for bond in mol.GetBonds()
        if bond.GetBondTypeAsDouble() == 1  # Single bond
        and bond.GetBeginAtom().GetAtomicNum() == 8  # Oxygen
        and bond.GetEndAtom().GetAtomicNum() == 1  # Hydrogen
        and bond.GetEndAtom().GetIsAromatic() == False  # Non-aromatic oxygen
    )


def fr_alkene(mol: Mol) -> int:
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
    new_df = df.copy()
    mols = new_df["MOL"]

    # Polarity and charge
    new_df["TPSA"] = mols.swifter.apply(
        rdMolDescriptors.CalcTPSA
    )  # Topological polar surface area
    # Aromatic rings
    new_df["n_arorings"] = mols.swifter.apply(rdMolDescriptors.CalcNumAromaticRings)

    return new_df


def expand_dataset(df: pd.DataFrame) -> pd.DataFrame:
    new_df = df.copy()
    new_df["MOL"] = new_df["SMILES"].swifter.apply(Chem.MolFromSmiles)
    new_df = new_df.drop(columns=["SMILES"])
    new_df = expand_structural_features(df)
    new_df = expand_physicochemical_features(new_df)
    new_df = expand_electronic_features(new_df)
    new_df = new_df.drop(columns=["MOL"])

    return new_df


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=["ACTIVE"]), df["ACTIVE"]


class BasePreprocessing:
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
    def __init__(self) -> None:
        self.columns: List[str] = None
        self.max_features: int = None
        self.base_transform: BasePreprocessing = BasePreprocessing()
        self.fitted_data: pd.DataFrame = None

    def fit_transform(
        self,
        df: pd.DataFrame,
        mode: Literal["nbest", "FPR", "FDR", "FWER"] = "nbest",
        max_features: int = 10,
        expand: bool = False,
    ) -> pd.DataFrame:
        new_df = self.base_transform.fit_transform(df, expand)

        match mode:
            case "nbest":
                top_features = SelectKBest(
                    score_func=mutual_info_classif, k=new_df.shape[0] - 1
                )
            case "FPR":
                top_features = SelectFpr(score_func=mutual_info_classif, alpha=1)
            case "FDR":
                top_features = SelectFdr(score_func=mutual_info_classif, alpha=1)
            case "FWER":
                top_features = SelectFwe(score_func=mutual_info_classif, alpha=1)

        X, y = split_features_target(new_df)
        top_features = top_features.fit(X, y)
        scores = pd.Series(top_features.scores_, index=top_features.feature_names_in_)

        self.columns = list(scores.sort_values(ascending=False).index)
        self.max_features = max_features
        self.fitted_data = new_df

        return new_df[self.columns[:max_features] + ["ACTIVE"]]

    def transform(self, feature_df: pd.DataFrame, expand: bool = False) -> pd.DataFrame:
        new_df = self.base_transform.transform(feature_df, expand)
        return new_df[self.columns[: self.max_features]]

    def change_fitted_data(self, max_features: int) -> pd.DataFrame:
        self.max_features = max_features
        return self.fitted_data[self.columns[:max_features] + ["ACTIVE"]]


def compute_score(selector: BaseEstimator):
    try:
        importance = np.absolute(selector.estimator_.coef_).squeeze()
    except AttributeError:
        importance = selector.estimator_.feature_importances_

    return importance


class FromModelPreprocessing:
    def __init__(self) -> None:
        self.base_transform: BasePreprocessing = BasePreprocessing()
        self.columns: List[str] = None
        self.max_features: int = None
        self.fitted_data: pd.DataFrame = None

    def fit_transform(
        self,
        df: pd.DataFrame,
        model: BaseEstimator,
        max_features: int = 10,
        expand: bool = False,
    ) -> pd.DataFrame:
        new_df = self.base_transform.fit_transform(df, expand)
        (
            X,
            y,
        ) = split_features_target(new_df)

        selector = SelectFromModel(model)
        selector = selector.fit(X, y)
        importance = compute_score(selector)

        scores = pd.Series(data=importance, index=selector.feature_names_in_.squeeze())

        self.columns = list(scores.sort_values(ascending=False).index)
        self.max_features = max_features
        self.fitted_data = new_df

        return new_df[self.columns[:max_features] + ["ACTIVE"]]

    def transform(self, feature_df: pd.DataFrame, expand: bool = False) -> pd.DataFrame:
        new_df = self.base_transform.transform(feature_df, expand)
        return new_df[self.columns[: self.max_features]]

    def change_fitted_data(self, max_features: int) -> pd.DataFrame:
        self.max_features = max_features
        return self.fitted_data[self.columns[:max_features] + ["ACTIVE"]]
