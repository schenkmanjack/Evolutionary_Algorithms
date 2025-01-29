import torch

# generate random slope and y-intercept values within bounds 
class LinearDataGenerator:
    def __init__(self, num_lines, num_points, slope_bounds = [-1, 1], y_intercept_bounds = [-1, 1], x_range = [-5, 5]):
        self.num_lines = num_lines
        self.slope_bounds = slope_bounds
        self.y_intercept_bounds = y_intercept_bounds
        self.x_range = x_range
        # generate slope and y-intercept values
        slope = torch.rand(num_lines) * (slope_bounds[1] - slope_bounds[0]) + slope_bounds[0]
        y_intercept = torch.rand(num_lines) * (y_intercept_bounds[1] - y_intercept_bounds[0]) + y_intercept_bounds[0]
        self.slope = slope
        self.y_intercept = y_intercept
        # generate data points
        self.x = torch.rand(num_points) * (x_range[1] - x_range[0]) + x_range[0]
        # repeat y intercept num_points times
        y_intercept = y_intercept.unsqueeze(1).repeat(1, num_points)
        self.y = slope.unsqueeze(1) @ self.x.unsqueeze(0) + y_intercept
        # max and min
        self.y_min = torch.min(self.y)
        self.y_max = torch.max(self.y)
    
    def get_y_bounds(self):
        return self.y_min, self.y_max
    
    def get_x_bounds(self):
        return self.x_range
    
    def get_data(self):
        return (self.x, self.y)



# NUM_LINES = 20
# NUM_POINTS = 100
# SLOPE_BOUNDS = [-1, 1]
# Y_INTERCEPT_BOUNDS = [-2, 2]
# X_RANGE = [-10, 10]

# data_generator = LinearDataGenerator(num_lines=NUM_LINES, num_points=NUM_POINTS, slope_bounds=SLOPE_BOUNDS, 
# y_intercept_bounds=Y_INTERCEPT_BOUNDS, x_range=X_RANGE)


