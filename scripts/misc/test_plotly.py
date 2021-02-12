import plotly.express as px

from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.validators.scatter.marker import SymbolValidator


def main():


    symbols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    # subplot
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Plot 1", "Plot 2"))

    
    fig.add_trace(
        go.Scatter(x=[1], y=[4], mode='markers', name='reg', marker_symbol=0, marker_color='red'),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=[1], y=[9], mode='markers', name='ft', marker_symbol=1, marker_color='blue'),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=[20], y=[50], mode='markers', name='reg', marker_symbol=0, marker_color='red', showlegend=False),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(x=[20], y=[100], mode='markers', name='ft', marker_symbol=1, marker_color='blue', showlegend=False),
        row=1, col=2
    )


    fig.update_layout(height=600, width=800, title_text="Side By Side Subplots")
    fig.show()



if __name__ == '__main__':
    main()