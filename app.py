import json
from datetime import datetime, timedelta
import jsonschema
import pandas as pd
import numpy as np
import concurrent.futures
import plotly.graph_objs as go
from jsonschema import validate, ValidationError
import urllib.parse
from multiprocessing import Pool
import dash
from dash import dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

# Constants
WORKING_DAYS_PER_SPRINT = 9

# Load Holidays
def load_holidays(filename):
    try:
        with open(filename, 'r') as file:
            holidays_data = json.load(file)

        if isinstance(holidays_data, dict):
            all_holidays = [
                {'country': country, **holiday}
                for country, holidays in holidays_data.items()
                for holiday in holidays
            ]
        else:
            all_holidays = holidays_data

        holidays_df = pd.DataFrame(all_holidays)
        holidays_df['date'] = pd.to_datetime(holidays_df['date'])
        return holidays_df.set_index('date')
    except (json.JSONDecodeError, pd.errors.EmptyDataError) as e:
        print(f"Error: {filename} is not a valid JSON file or empty: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Unexpected error: {e}")
        return pd.DataFrame()

# JSON Schema for squad data validation
SQUAD_DATA_SCHEMA = {
    "type": "object",
    "properties": {
        "squads": {
            "type": "object",
            "patternProperties": {
                "^[A-Za-z0-9_-]+$": {
                    "type": "object",
                    "properties": {
                        "members": {
                            "type": "object",
                            "patternProperties": {
                                "^[A-Za-z0-9_-]+$": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                }
                            }
                        },
                        "sprint_data": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "start_date": {
                                        "type": "string",
                                        "format": "date"
                                    },
                                    "stories_completed": {
                                        "type": "integer",
                                        "minimum": 0
                                    }
                                },
                                "required": ["start_date", "stories_completed"]
                            }
                        }
                    },
                    "required": ["members", "sprint_data"]
                }
            }
        },
        "estimated_stories": {
            "type": "integer",
            "minimum": 0
        },
        "sprint_start_date": {
            "type": "string",
            "format": "date"
        },
        "num_simulations": {
            "type": "integer",
            "minimum": 1
        }
    },
    "required": ["squads", "estimated_stories", "sprint_start_date", "num_simulations"]
}

# Validate JSON against the schema
def validate_json(json_data):
    try:
        json_data_dict = json.loads(json_data)
        jsonschema.validate(json_data_dict, SQUAD_DATA_SCHEMA)
    except json.JSONDecodeError as e:
        return f"JSON Syntax Error: {str(e)}"
    except jsonschema.ValidationError as e:
        return f"JSON Validation Error: {e.message}"
    return None  # No validation errors

# Adjust Holidays for Weekends
def adjust_for_weekend(holidays):
    return [
        holiday - timedelta(days=1) if holiday.weekday() == 5 else
        holiday + timedelta(days=1) if holiday.weekday() == 6 else
        holiday
        for holiday in holidays
    ]

# Check for Holidays and Absences
def is_holiday(date, holidays):
    return date in holidays

def is_member_absent(date, member_absences):
    date_str = date.strftime("%Y-%m-%d")  # Convert the date to a string
    for period in member_absences:
        if isinstance(period, list):
            start_date = datetime.strptime(period[0], "%Y-%m-%d")
            end_date = datetime.strptime(period[1], "%Y-%m-%d")
            if start_date <= date <= end_date:
                return True
        elif isinstance(period, str):
            if date_str == period:  # Compare the date as a string
                return True
    return False

# Calculate Working Days for a Squad
def calculate_working_days_for_squad(start_date, squad_members, us_holidays, india_holidays, member_absences):
    end_date = start_date + timedelta(days=WORKING_DAYS_PER_SPRINT)
    us_holidays_set = set(us_holidays.index)
    india_holidays_set = set(india_holidays.index)
    business_days = pd.bdate_range(start_date, end_date).difference(us_holidays_set.union(india_holidays_set))
    
    total_working_days = sum(
        1
        for single_date in business_days
        if not any(is_member_absent(single_date, member_absences.get(member, [])) for member in squad_members)
    )
    return total_working_days

# Monte Carlo simulation for a single squad
def simulate_squad(squad, estimated_stories, us_holidays, india_holidays, num_simulations):
    velocities = []
    member_absences = squad.get("absences", {})
    total_members = sum(len(members) for members in squad["members"].values())

    for sprint_entry in squad["sprint_data"]:
        start_date = datetime.strptime(sprint_entry["start_date"], "%Y-%m-%d")
        stories = sprint_entry["stories_completed"]
        working_days = calculate_working_days_for_squad(start_date, squad["members"], us_holidays, india_holidays, member_absences)
        adjusted_stories = stories
        velocities.append(adjusted_stories / working_days if working_days else 0)

    average_velocity = np.mean(velocities)
    std_dev_velocity = np.std(velocities)

    sprints_needed = []
    for _ in range(num_simulations):
        total_stories = 0
        sprints = 0
        while total_stories < estimated_stories:
            future_sprint_start = datetime.now() + timedelta(days=sprints * (WORKING_DAYS_PER_SPRINT + 1))
            working_days = calculate_working_days_for_squad(future_sprint_start, squad["members"], us_holidays, india_holidays, member_absences)
            stories_completed = np.random.normal(average_velocity, std_dev_velocity) * working_days
            stories_completed = np.round(stories_completed)
            total_stories += stories_completed
            sprints += 1
        sprints_needed.append(sprints)

    return np.array(sprints_needed)

# Monte Carlo simulation for multiple squads using ThreadPoolExecutor
def monte_carlo_simulation_main(squads_data, estimated_stories, us_holidays, india_holidays, num_simulations=10000):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        args = [(squad, estimated_stories, us_holidays, india_holidays, num_simulations) for squad in squads_data.values()]
        results = list(executor.map(lambda args: simulate_squad(*args), args))
    return dict(zip(squads_data.keys(), results))

# Load the JSON data
try:
    with open('squad_data.json', 'r') as file:
        squads_data = json.load(file)
except (FileNotFoundError, json.JSONDecodeError) as e:
    print(f"Error loading 'squad_data.json': {e}")
    squads_data = {'squads': {}, 'estimated_stories': 0, 'sprint_start_date': datetime.now().strftime("%Y-%m-%d"), 'num_simulations': 1000}

# JSON Schema for Squad Data
squad_data_schema = {
    "type": "object",
    "properties": {
        "squads": {"type": "object"},
        "estimated_stories": {"type": "number"},
        "sprint_start_date": {"type": "string", "format": "date"},
        "num_simulations": {"type": "number"},
    },
    "required": ["squads", "estimated_stories", "sprint_start_date", "num_simulations"],
}

# JSON validator function
def validate_json(data):
    try:
        parsed_data = json.loads(data)
        validate(parsed_data, squad_data_schema)
        return None
    except json.JSONDecodeError as e:
        return f"Invalid JSON data: {str(e)}"
    except ValidationError as e:
        return f"Invalid Squad Data: {str(e)}"

# Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# App layout
app.layout = dbc.Container([
    html.H1("Sprint Forecasting Dashboard"),
    dbc.Row([
        dbc.Col([
            html.Label("Select Squad(s):"),
            dcc.Dropdown(
                id="squad-dropdown",
                options=[{'label': squad, 'value': squad} for squad in squads_data['squads'].keys()],
                multi=True,
                value=[list(squads_data['squads'].keys())[0]]
            ),
        ]),
        dbc.Col([
            html.Label("Estimated Stories:"),
            dcc.Input(
                id='estimated-stories',
                type='number',
                value=squads_data['estimated_stories']
            ),
        ]),
        dbc.Col([
            html.Label("Select Simulation Parameters:"),
            dcc.DatePickerSingle(
                id='sprint-start-date',
                display_format='YYYY-MM-DD',
                date=squads_data['sprint_start_date']
            ),
        ]),
        dbc.Col([
            html.Label("Number of Simulations:"),
            dcc.Input(
                id='num-simulations',
                type='number',
                value=squads_data['num_simulations']
            ),
        ]),
    ]),
    html.Br(),
    dbc.Row([
        dbc.Col([
            dbc.Button("Run Simulation", id="run-button", color="primary"),
        ]),
        dbc.Col([
            dbc.Button("Edit Squad Data", id="edit-squad-button", color="info"),
            dbc.Modal([
                dbc.ModalHeader("Edit Squad Data"),
                dbc.ModalBody([
                    dcc.Textarea(
                        id="json-data-input",
                        style={'width': '100%', 'height': '300px'},
                        value=json.dumps(squads_data['squads'], indent=4),
                    ),
                    html.Div(id="save-status"),
                ]),
                dbc.ModalFooter([
                    dbc.Button("Save Changes", id="save-button", color="primary"),
                    dbc.Button("Close", id="close-button", color="danger"),
                ]),
            ], id="squad-data-modal", is_open=False),
        ]),
        dbc.Col([
            dbc.Button("Export to CSV", id="export-csv-button", color="success"),
            dcc.Download(id="download-data"),
        ]),
    ]),
    dcc.Graph(
        id='output-graph',
        config={'displayModeBar': False}
    ),
    html.Div(id='sprint-results')
])

# Callback to run the simulation and update the graph
@app.callback(
    [Output('output-graph', 'figure'),
     Output('sprint-results', 'children')],
    [Input('run-button', 'n_clicks')],
    [State('squad-dropdown', 'value'),
     State('estimated-stories', 'value'),
     State('sprint-start-date', 'date'),
     State('num-simulations', 'value')]
)
def run_simulation(n_clicks, selected_squads, estimated_stories, sprint_start_date, num_simulations):
    if n_clicks is None:
        raise PreventUpdate

    squads_data_copy = squads_data.copy()
    squads_data_copy['estimated_stories'] = estimated_stories
    squads_data_copy['sprint_start_date'] = sprint_start_date
    squads_data_copy['num_simulations'] = num_simulations

    us_holidays = load_holidays('us_holidays.json')
    india_holidays = load_holidays('india_holidays.json')

    results = monte_carlo_simulation_main(
        {squad: squads_data_copy['squads'][squad] for squad in selected_squads},
        estimated_stories,
        us_holidays,
        india_holidays,
        num_simulations
    )

    # Create traces for each selected squad
    traces = []
    stats_text = []
    for squad_name in selected_squads:
        selected_squad_results = results[squad_name]

        # Calculate mean outside of the conditional block
        avg_sprints_needed = np.mean(selected_squad_results)
        median_sprints_needed = np.median(selected_squad_results)
        std_dev_sprints_needed = np.std(selected_squad_results)
        upper_bound = avg_sprints_needed + 2 * std_dev_sprints_needed
        lower_bound = avg_sprints_needed - 2 * std_dev_sprints_needed
        percentile_80 = np.percentile(selected_squad_results, 80)
        percentile_95 = np.percentile(selected_squad_results, 95)

        # Filter outliers
        filtered_results = [
            sprints
            for sprints in selected_squad_results
            if lower_bound <= sprints <= upper_bound
        ]

        # Create histogram trace
        trace = go.Histogram(
            x=filtered_results,
            xbins=dict(
                start=min(filtered_results),
                end=max(filtered_results),
                size=1
            ),
            opacity=0.75,
            name=squad_name
        )

        traces.append(trace)

        # Generate simulation results text
        stats = html.Div([
            html.H3(f'{squad_name} Simulation Results for {num_simulations} Simulations'),
            html.P(f'Mean Sprints Needed: {avg_sprints_needed:.2f}'),
            html.P(f'Median Sprints Needed: {median_sprints_needed:.2f}'),
            html.P(f'Standard Deviation: {std_dev_sprints_needed:.2f}'),
            html.P(f'95% Confidence Interval: ({lower_bound:.2f}, {upper_bound:.2f})'),
            html.P(f"80th Percentile: {percentile_80}"),
            html.P(f"95th Percentile: {percentile_95}"),
            html.Br(),
        ])
        stats_text.append(stats)

    # Create layout
    layout = go.Layout(
        title='Sprint Forecasting for Selected Squads',
        xaxis=dict(title='Sprints Needed'),
        yaxis=dict(title='Frequency'),
        bargap=0.1,
        barmode='overlay',
        legend=dict(orientation="h")
    )

    # Create figure
    figure = go.Figure(data=traces, layout=layout)

    return figure, stats_text

# Callback to open/close the edit squad data modal
@app.callback(
    [Output('squad-data-modal', 'is_open'),
     Output('json-data-input', 'value')],
    [Input('edit-squad-button', 'n_clicks'),
     Input('save-button', 'n_clicks'),
     Input('close-button', 'n_clicks')],
    [State('squad-data-modal', 'is_open')]
)
def toggle_modal(edit_clicks, save_clicks, close_clicks, is_open):
    ctx = dash.callback_context
    if ctx.triggered_id == 'edit-squad-button':
        return not is_open, json.dumps(squads_data['squads'], indent=4)
    elif ctx.triggered_id == 'save-button':
        new_data = ctx.inputs['json-data-input.value']
        validation_error = validate_json(new_data)
        if validation_error:
            return is_open, new_data
        else:
            squads_data['squads'] = json.loads(new_data)
            with open('squad_data.json', 'w') as file:
                json.dump(squads_data, file, indent=4)
            return not is_open, json.dumps(squads_data['squads'], indent=4)
    elif ctx.triggered_id == 'close-button':
        return not is_open, json.dumps(squads_data['squads'], indent=4)
    return is_open, json.dumps(squads_data['squads'], indent=4)

# Callback to display save status
@app.callback(
    Output('save-status', 'children'),
    [Input('save-button', 'n_clicks')]
)
def display_save_status(n_clicks):
    if n_clicks:
        return "Changes saved successfully!"
    return ""

# Callback to export data to CSV
@app.callback(
    Output('download-data', 'data'),
    [Input('export-csv-button', 'n_clicks')],
    [State('squad-dropdown', 'value')]
)
def export_to_csv(n_clicks, selected_squads):
    if n_clicks:
        results = monte_carlo_simulation_main(
            {squad: squads_data['squads'][squad] for squad in selected_squads},
            squads_data['estimated_stories'],
            load_holidays('us_holidays.json'),
            load_holidays('india_holidays.json'),
            squads_data['num_simulations']
        )
        data = []
        for squad_name, squad_results in results.items():
            data.append({'Squad': squad_name, 'Mean Sprints Needed': np.mean(squad_results),
                         'Median Sprints Needed': np.median(squad_results),
                         'Standard Deviation': np.std(squad_results),
                         '80th Percentile': np.percentile(squad_results, 80),
                         '95th Percentile': np.percentile(squad_results, 95)})

        df = pd.DataFrame(data)
        csv_string = df.to_csv(index=False, encoding='utf-8')
        csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)
        return dict(content=csv_string, filename="sprint_results.csv")

if __name__ == '__main__':
    app.run_server(debug=True)
