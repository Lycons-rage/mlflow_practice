import mlflow

def calc_sum(x,y):
    return x+y

if __name__ == "__main__":
    # we can track this experiment using mlflow
    # starting mlflow server
    with mlflow.start_run():
        x,y = 12,17
        z = calc_sum(x,y)
        # tracking the experiment with the mlflow
        mlflow.log_param("X",x) # tracking param
        mlflow.log_param("Y",y)
        mlflow.log_metric("Z",z) # tracking output


