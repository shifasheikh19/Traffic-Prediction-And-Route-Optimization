import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
import matplotlib.pyplot as plt


merged_data = pd.read_csv('cleaned_merged_data_fixed.csv')


features = ['freeFlowSpeed', 'temp', 'feels_like', 'temp_freeFlowSpeed', 'feels_like_freeFlowSpeed'] + \
           [col for col in merged_data.columns if col.startswith('weather_main_')]


target = 'currentSpeed'


merged_data.drop_duplicates(inplace=True)


merged_data = merged_data.dropna(subset=[target])


X = merged_data[features]
y = merged_data[target]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)


preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), features)
    ])


model = GradientBoostingRegressor(random_state=42)

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', model)])
pipeline.fit(X_train, y_train)


y_pred = pipeline.predict(X_test)


print("Gradient Boosting - Actual Values:", y_test.values)
print("Gradient Boosting - Predictions:", y_pred)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Actual Speeds')
plt.ylabel('Predicted Speeds')
plt.title('Predicted vs. Actual Speeds')
plt.grid(True)
plt.show()

route_segments = [(0, 1), (1, 2), (2, 3)]
predicted_traffic = {}

for segment in route_segments:

    segment_features = np.array(
        [[70, 25, 27, 70 * 25, 70 * 27] + [0] * (len(features) - 5)])


    segment_features_df = pd.DataFrame(segment_features, columns=features)


    predicted_speed = pipeline.predict(segment_features_df)


    predicted_traffic[segment] = 1 / predicted_speed[0]





def create_data_model(predicted_traffic):

    data = {}


    base_distance_matrix = [
        [0, 29, 20, 21],
        [29, 0, 15, 17],
        [20, 15, 0, 28],
        [21, 17, 28, 0],
    ]


    adjusted_distance_matrix = [
        [0] * len(base_distance_matrix) for _ in range(len(base_distance_matrix))
    ]

    for i in range(len(base_distance_matrix)):
        for j in range(len(base_distance_matrix[i])):
            if (i, j) in predicted_traffic:
                adjusted_distance_matrix[i][j] = int(base_distance_matrix[i][j] * predicted_traffic[(i, j)])
            else:
                adjusted_distance_matrix[i][j] = base_distance_matrix[i][j]

    data['distance_matrix'] = adjusted_distance_matrix
    data['num_vehicles'] = 1
    data['depot'] = 0
    return data

def solve_vrp_with_adjusted_costs(predicted_traffic):

    data = create_data_model(predicted_traffic)
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)

    def adjusted_cost_callback(from_index, to_index):

        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)

        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(adjusted_cost_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        print_solution(manager, routing, solution)


        base_cost = calculate_total_cost(np.array([
            [0, 29, 20, 21],
            [29, 0, 15, 17],
            [20, 15, 0, 28],
            [21, 17, 28, 0],
        ]), [0, 1, 2, 3, 0])

        adjusted_cost = calculate_total_cost(np.array(data['distance_matrix']), [0, 1, 2, 3, 0])

        plt.figure(figsize=(10, 6))
        plt.bar(['Base Cost', 'Adjusted Cost'], [base_cost, adjusted_cost], color=['green', 'orange'])
        plt.ylabel('Total Cost')
        plt.title('Cost Comparison Before and After Traffic Adjustment')
        plt.grid(True)
        plt.show()

def calculate_total_cost(matrix, route):
    return sum(matrix[route[i], route[i + 1]] for i in range(len(route) - 1))

def print_solution(manager, routing, solution):
    print('Objective: {} cost units'.format(solution.ObjectiveValue()))
    index = routing.Start(0)
    plan_output = 'Route:\n'
    while not routing.IsEnd(index):
        plan_output += ' {} ->'.format(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))
    plan_output += ' {}\n'.format(manager.IndexToNode(index))
    print(plan_output)

if __name__ == '__main__':
    solve_vrp_with_adjusted_costs(predicted_traffic)

