from typing import Any, Dict, Tuple
import pandas as pd
from .utils import get_number_order

def confidence_region_as_latex(central: float, err: float) -> str:
    precision = int(-get_number_order(err))  # type: ignore
    repr_str = f'${central:.{precision}f} \\pm {err:.{precision}f}$'
    return repr_str


def mean_rms_confidence_region(data: pd.Series) -> Tuple[str, str]:
    mean = data.mean()
    rms_err = data.std()
    precision = int(-get_number_order(err))
    mean_str = f"{central:.{precision}f}"
    rms_str = f"{err:.{precision}f}"
    return mean_str, rms_str

def mean_rms_confidence_region_str(data: Any, rms_factor: float = 1.,) -> str:
    mean = data.mean()
    rms_err = data.std()
    repr_str = confidence_region_as_latex(mean, rms_factor*rms_err)
    return repr_str

def confidence_region_df_as_latex(data: pd.DataFrame,
                         groupby: str = None,
                         sort_values: Dict[str, Any] = {},
                         keep_index: bool = True) -> str:
    if groupby:
        confidence_df = data \
            .groupby(groupby) \
            .agg(mean_rms_confidence_region_str)
    else:
        confidence_df = data \
            .apply(mean_rms_confidence_region_str, axis=0) \
            .to_frame()
    
    if sort_values:
        confidence_df = confidence_df.sort_values(**sort_values)
     
    if keep_index:
        confidence_df = confidence_df.reset_index()
    else:
        confidence_df = confidecne_df.reset_index(drop=True)
    column_format = f"||{'|'.join((1+len(confidence_df.columns))*['c'])}||"
    df_as_latex = confidence_df.style.to_latex(
        column_format=column_format,
        clines='all;data'
    )
    return df_as_latex