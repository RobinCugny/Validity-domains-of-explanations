import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from dash import Dash,html,dcc, Input, Output
import numpy as np

data=pd.read_csv("data/flame.txt",sep="\t",header=None)
data.columns = ["x0","x1","label"]
Z = pickle.load(open("data/inter_v2/flame_dists_exp.p", "rb"))
exp_X = pickle.load(open("data/inter_v2/flame_exp.p", "rb"))
x_0="x0"
x_1="x1"
label="label"
feature_names = [x_0,x_1]

# data=pd.read_csv("data/pima/biased_pima.csv")
# Z = pickle.load(open("data/pima/Z.p", "rb"))
# xx = pickle.load(open("data/pima/xx.p", "rb"))
# yy = pickle.load(open("data/pima/yy.p", "rb"))
# exp_X = pickle.load(open("data/pima/exp_X.p", "rb"))
# x_0="Glucose"
# x_1="BloodPressure"
# feature_names=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
#        'BMI', 'DiabetesPedigreeFunction', 'Age']
# label="Outcome"

app = Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='graph',style={'width': '120vh', 'height': '90vh'}),
    dcc.Graph(id='graph-exp',style={'width': '120vh', 'height': '90vh'}),
])


@app.callback(
    Output('graph', 'figure'),
    Output('graph-exp', 'figure'),
    Input('graph', 'clickData'))
def update_figure(input_value):
    if input_value is not None:
        idx=input_value['points'][0]['pointNumber']
        print("\n")
        print(data.iloc[idx])
        print("explanation",exp_X.values[idx])
    else:
        idx=0

    # col = data[label].copy()
    # col[idx]=3
    col = np.asarray(Z[idx]<0.1).astype(int)+1
    col[idx]=0
    # col = np.asarray(Z[idx])
    
    symbols = ['circle' if y==1 else 'square' for y in data['label']]
    trace2 = go.Scatter(x=data[x_0],
            y=data[x_1],
            mode='markers',
            marker=dict(size=10,color=col),
            marker_line_width=2,
            marker_symbol=symbols,
            text=exp_X.values
            )
    fig = make_subplots()
    fig.add_trace(trace2)
    fig.update_layout(
    title="Validity domain of explanations",
    xaxis_title=x_0,
    yaxis_title=x_1,
)
    order = np.argsort(np.abs(exp_X.values[idx]))
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        y=np.asarray(feature_names)[order],
        x=np.asarray(exp_X.values[idx])[order],
        name='Explanation',
        orientation='h',
        marker=dict(
        color=np.asarray(exp_X.values[idx])[order]
        )))
    fig2.update_layout(
    title="Selected explanation",
    xaxis_title="Shap value",
    yaxis_title="features",
)

    return fig,fig2

if __name__ == '__main__':
    app.run_server(debug=True)