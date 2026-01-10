import dash
from dash import html

app = dash.Dash(__name__)
app.layout = html.Div("Dash is running!")
server = app.server

if __name__ == "__main__":
    print("Starting minimal Dash...")
    app.run(host="127.0.0.1", port=8050, debug=True)
