from typing import Union
from pathlib import Path
import pandas as pd

# TODO Move this module to Archai utils stuff

def get_search_csv(output_path: Union[str, Path], iteration_num: int = -1) -> pd.DataFrame:
    """Reads the search csv file from the output path and returns a pandas dataframe

    Args:
        output_path (Union[str, Path]): Path to the output directory
        iteration_num (int, optional): Search iteration to read from. Defaults to -1, which will point to the last iteration

    Returns:
        pd.DataFrame: Pandas dataframe with the search state
    """
    if iteration_num == -1:
        search_csv_path = max(Path(output_path).glob("search_state_*.csv"), key=lambda x: int(x.stem.split("_")[-1]))
    else:
        search_csv_path = Path(output_path) / f"search_state_{iteration_num}.csv"

    if not search_csv_path.is_file():
        raise FileNotFoundError(f"Search csv file not found at {search_csv_path}")

    df = pd.read_csv(search_csv_path)
    return df

def get_csv_as_stylized_html(df: pd.DataFrame) -> str:
    styled_table = df.style.set_properties(**{'background-color': 'lightblue',
                                            'color': 'black',
                                            'border-color': 'white',
                                            'font-size': '12pt'
                                            })

    html_table = styled_table.to_html()
    html_with_scrollbar = f'<div style="height: 300px; overflow-y: scroll;">{html_table}</div>'

    return html_with_scrollbar