# austiezr_lambdata

austiezr_lambdata is a Python library created to provide tools to aid in Lambda School Data Science Curriculum.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install austiezr_lambdata.

```bash
pip install --index-url https://test.pypi.org/simple/ austiezr-lambdata
```

## Usage

```python
from austiezr_lambdata.austiezr_lambdata import MVP, TransformDF

TransformDF.add_to_df(new_list, df) # appends list as new column
TransformDF.date_split(df, date_col) # converts date column to datetime, splits into component columns
MVP(model).fast_first(df, target) # returns baselines and metrics for fast first modeling
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)