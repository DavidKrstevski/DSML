"""Top-level workflow orchestration."""

from .data_loading import load_dataset
from .visualizations import create_visualizations
from .modeling import run_full_analysis


def main():
    """Load data, create figures and run modeling."""
    df, corr = load_dataset()
    create_visualizations(df, corr)
    run_full_analysis(df)


if __name__ == "__main__":
    main()
